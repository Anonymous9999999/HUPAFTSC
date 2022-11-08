import torch
import hashlib
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as pl
from ..loss_functions import MaskedMSELoss
from ...utils import check_nan
from ...utils import padding_mask
from .pretrain_hp_config import PretrainHPConfig


class MVTSHP(PretrainHPConfig):
    def __init__(self, is_grid_search):
        default_hp = {'masking_ratio': 0.45,
                      'mean_mask_length': 5,
                      'mask_mode': None,
                      'distribution': 'geometric'
                      }
        hp_range = {'masking_ratio': (0.15, 0.3, 0.45),
                    'mean_mask_length': (3, 5, 7),
                    'mask_mode': ('separate', None),
                    'distribution': ('geometric', None)
                    }
        super(MVTSHP, self).__init__(default_hp, hp_range, is_grid_search=is_grid_search)


def _geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked

    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (
            1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


def _noise_mask(X, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = _geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(_geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1),
                           X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

    return mask


def _compensate_masking(X, mask):
    """
    Compensate feature vectors after masking values, in a way that the matrix product W @ X would not be affected on average.
    If p is the proportion of unmasked (active) elements, X' = X / p = X * feat_dim/num_active
    Args:
        X: (batch_size, seq_length, feat_dim) torch tensor
        mask: (batch_size, seq_length, feat_dim) torch tensor: 0s means mask and predict, 1s: unaffected (active) input
    Returns:
        (batch_size, seq_length, feat_dim) compensated features
    """

    # number of unmasked elements of feature vector for each time step
    num_active = torch.sum(mask, dim=-1).unsqueeze(-1)  # (batch_size, seq_length, 1)
    # to avoid division by 0, set the minimum to 1
    num_active = torch.max(num_active, torch.ones(num_active.shape, dtype=torch.int16))  # (batch_size, seq_length, 1)
    return X.shape[-1] * X / num_active


class MVTS:
    def __init__(self):
        pass

    @staticmethod
    def collate_superv(data):
        raise NotImplementedError

    @staticmethod
    def collate_unsuperv(data, max_len=None, mask_compensation=False):
        """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
        Args:
            data: len(batch_size) list of tuples (X, mask).
                - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
                - mask: boolean torch tensor of shape (seq_length, feat_dim); variable seq_length.
            max_len: global fixed sequence length. Used for architectures requiring fixed length input,
                where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
        Returns:
            X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
            targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
            target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
                0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
            padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 ignore (padding)
        """

        batch_size = len(data)
        features, masks = zip(*data)

        # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
        lengths = [X.shape[0] for X in features]  # original sequence length for each time series
        if max_len is None:
            max_len = max(lengths)
        X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
        target_masks = torch.zeros_like(X,
                                        dtype=torch.bool)  # (batch_size, padded_length, feat_dim) masks related to objective
        for i in range(batch_size):
            end = min(lengths[i], max_len)
            X[i, :end, :] = features[i][:end, :]
            target_masks[i, :end, :] = masks[i][:end, :]

        targets = X.clone()
        X = X * target_masks  # mask input
        if mask_compensation:
            X = _compensate_masking(X, target_masks)

        padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                     max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep
        target_masks = ~target_masks  # inverse logic: 0 now means ignore, 1 means predict
        return X, targets, target_masks, padding_masks

    @staticmethod
    def pretrain_dataloader(train_X,
                            val_X,
                            batch_size,
                            mean_mask_length: int = 3,
                            masking_ratio: float = 0.15,
                            mask_mode: str = 'separate',
                            distribution: str = 'geometric',
                            mask_compensation: bool = False):
        train_dataset = MVTSImputationDataset(train_X,
                                              mean_mask_length=mean_mask_length,
                                              masking_ratio=masking_ratio,
                                              mask_mode=mask_mode,
                                              distribution=distribution
                                              )
        val_dataset = MVTSImputationDataset(val_X,
                                            mean_mask_length=mean_mask_length,
                                            masking_ratio=masking_ratio,
                                            mask_mode=mask_mode,
                                            distribution=distribution
                                            )
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True,
                                  collate_fn=lambda x: MVTS.collate_unsuperv(x, mask_compensation=mask_compensation))
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                collate_fn=lambda x: MVTS.collate_unsuperv(x, mask_compensation=mask_compensation))
        return (train_loader, val_loader)


class MVTSImputationDataset(Dataset):
    def __init__(self,
                 sktime_data: np.ndarray,
                 mean_mask_length: int = 3,
                 masking_ratio: float = 0.15,
                 mask_mode: str = 'separate',
                 distribution: str = 'geometric'
                 ):
        self.sktime_data = sktime_data
        self.mean_mask_length = mean_mask_length
        self.masking_ratio = masking_ratio
        self.mask_mode = mask_mode
        self.distribution = distribution

    @property
    def hash_value(self):
        return hashlib.md5(str(self.sktime_data).encode('utf-8')).hexdigest()

    def __getitem__(self, ind):
        X = self.sktime_data[ind]
        mask = _noise_mask(X, self.masking_ratio, self.mean_mask_length, self.mask_mode, self.distribution)
        return torch.from_numpy(X), torch.from_numpy(mask)

    def __len__(self):
        return len(self.sktime_data)


class MvtsPretrainWrapper(pl.LightningModule):

    def __init__(self,
                 encoder,
                 feat_dim,
                 pretrain_viewer=None,
                 hp=None):
        super(MvtsPretrainWrapper, self).__init__()
        self.encoder = encoder
        self.pretrain_head = nn.Linear(encoder.output_dim, feat_dim)
        self.loss_func = MaskedMSELoss(reduction='none')
        self.pretrain_viewer = pretrain_viewer
        if self.pretrain_viewer is not None:
            self.pretrain_viewer.get_hp(hp)
        self.hp = hp

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        output = self.encoder.forward(X, padding_masks)
        output = self.pretrain_head(output)  # (batch_size, seq_length, feat_dim)
        return output

    def compute_loss(self, batch_data, stage='train'):
        X, targets, target_masks, padding_masks = batch_data

        predictions = self.forward(X, padding_masks)  # (batch_size, padded_length, feat_dim)

        # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
        target_masks = target_masks * padding_masks.unsqueeze(-1)

        # (num_active,) individual loss (square error per element) for each active value in batch

        loss = self.loss_func(predictions, targets, target_masks)
        if loss.shape[0] == 0:
            batch_loss, mean_loss = None, None
            mean_loss_value = None
        else:
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization
            mean_loss_value = mean_loss.item()

        if self.pretrain_viewer is not None:
            with torch.no_grad():
                self.pretrain_viewer.collect_mvts_data(self.current_epoch,
                                                       predictions,
                                                       targets,
                                                       target_masks,
                                                       mean_loss_value,
                                                       stage)

        return loss, batch_loss, mean_loss

    def training_step(self, train_batch, batch_idx):
        check_nan(train_batch, self)
        loss, batch_loss, mean_loss = self.compute_loss(train_batch, stage='train')
        if mean_loss is None:
            return None
        else:
            self.log('train_loss', mean_loss, prog_bar=True, on_step=False, on_epoch=True)
            return mean_loss

    def validation_step(self, val_batch, batch_idx):
        check_nan(val_batch, self)
        loss, batch_loss, mean_loss = self.compute_loss(val_batch, stage='val')
        if mean_loss is None:
            return None
        else:
            self.log('val_loss', mean_loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
