import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from ..loss_functions import HierarchicalContrastiveLoss
from ...utils import check_nan
from ...utils import get_X_padding_masks
from .pretrain_method_utils import TsGeneralPretrainDataset
from .pretrain_hp_config import PretrainHPConfig


class Ts2VecHP(PretrainHPConfig):
    def __init__(self, is_grid_search):
        default_hp = {'alpha': 0.5,
                      'mask_type': 'continuous'
                      }
        hp_range = {'alpha': [0, 0.3, 0.5, 0.7],
                    'mask_type': ['binomial', 'continuous', 'mask_last']}
        super(Ts2VecHP, self).__init__(default_hp, hp_range, is_grid_search=is_grid_search)


class Ts2Vec:
    def __init__(self):
        pass

    @staticmethod
    def collate_unsuperv(X):
        # (batch_size, seq_len, feat_dimension)
        X = torch.stack(X).float()
        padding_masks, lengths = get_X_padding_masks(X)
        return X, padding_masks

    @staticmethod
    def pretrain_dataloader(train_X, val_X, batch_size):
        train_dataset = TsGeneralPretrainDataset(train_X)
        val_dataset = TsGeneralPretrainDataset(val_X)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True,
                                  collate_fn=lambda x: Ts2Vec.collate_unsuperv(x))
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                collate_fn=lambda x: Ts2Vec.collate_unsuperv(x))
        return (train_loader, val_loader)


class Ts2VecPretrainWrapper(pl.LightningModule):
    """
    Time-Series Representation Learning via Temporal and Contextual Contrasting
    https://www.ijcai.org/proceedings/2021/0324.pdf
    """

    def __init__(self,
                 batch_size,
                 encoder,
                 project_dim=32,
                 alpha=0.5,
                 temporal_unit=0,
                 mask_type='binomial',
                 pretrain_viewer=None,
                 hp=None
                 ):
        super(Ts2VecPretrainWrapper, self).__init__()
        self.encoder = encoder
        self.batch_size = batch_size
        self.pretrain_head = nn.Linear(encoder.output_dim, project_dim)
        self.temporal_unit = temporal_unit

        # valid mask types: binomial, continuous, all_true, all_false, mask_last
        self.mask_type = mask_type
        self.h_contrastive_loss = HierarchicalContrastiveLoss(alpha=alpha, temporal_unit=temporal_unit)

        self.pretrain_viewer = pretrain_viewer
        if self.pretrain_viewer is not None:
            self.pretrain_viewer.get_hp(hp)
        self.hp = hp

    def forward(self, X, padding_masks=None, is_permute=False):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, feat_dim)
        """
        if is_permute:
            X = X.permute(0, 2, 1)
        if padding_masks is None:
            padding_masks = torch.ones((X.shape[0], X.shape[1]), dtype=torch.bool).to(X.device)
        output = self.encoder.forward(X, padding_masks)
        output = self.pretrain_head(output)  # (batch_size, seq_length, feat_dim)
        return output

    def _generate_continuous_mask(self, B, T, n=5, l=0.1):
        res = torch.full((B, T), True, dtype=torch.bool)
        if isinstance(n, float):
            n = int(n * T)
        n = max(min(n, T // 2), 1)

        if isinstance(l, float):
            l = int(l * T)
        l = max(l, 1)

        for i in range(B):
            for _ in range(n):
                t = np.random.randint(T - l + 1)
                res[i, t:t + l] = False
        return res

    def _generate_binomial_mask(self, B, T, p=0.5):
        return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

    def _take_per_row(self, A, indx, num_elem):
        all_indx = indx[:, None] + np.arange(num_elem)
        return A[torch.arange(all_indx.shape[0])[:, None], all_indx]

    def _mask_encoder_input(self, x, mask_type):
        # generate & apply mask
        if mask_type == 'binomial':
            mask = self._generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask_type == 'continuous':
            mask = self._generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask_type == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask_type == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask_type == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        else:
            raise NotImplementedError

        x[~mask] = 0
        return x

    def compute_loss(self, batch_data):
        X, padding_masks = batch_data

        # --------------------------------------------------------------------------------------------------------------
        # Generate two subsequences with intersection
        # --------------------------------------------------------------------------------------------------------------
        ts_l = X.size(1)
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=X.size(0))

        # The following aug1, aug2's seq_len may not be equal
        aug1 = self._take_per_row(X, crop_offset + crop_eleft,
                                  crop_right - crop_eleft)  # batch_size, seq_len, input_dim
        aug2 = self._take_per_row(X, crop_offset + crop_left, crop_eright - crop_left)  # batch_size, seq_len, input_dim
        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        # Do MASK on the generated subsequence
        # --------------------------------------------------------------------------------------------------------------
        aug1 = self._mask_encoder_input(aug1, self.mask_type)
        aug2 = self._mask_encoder_input(aug2, self.mask_type)
        # --------------------------------------------------------------------------------------------------------------

        # Generate a representation of the two subsequences
        # out: (batch_size, seq_len, project_dim)，Here the seq_len is shorter than the above seq_len,
        # which is the intersection of two overlap sequences
        out1, out2 = self.forward(aug1)[:, -crop_l:], self.forward(aug2)[:, :crop_l]
        loss = self.h_contrastive_loss(
            out1,
            out2
        )

        if self.pretrain_viewer is not None:
            with torch.no_grad():
                self.pretrain_viewer.collect_ts2vec_data(self.current_epoch)

        return loss, loss, loss

    def training_step(self, train_batch, batch_idx):
        check_nan(train_batch, self)
        loss, batch_loss, mean_loss = self.compute_loss(train_batch)
        if mean_loss is None:
            return None
        else:
            self.log('train_loss', mean_loss, prog_bar=True, on_step=False, on_epoch=True)
            return mean_loss

    def validation_step(self, val_batch, batch_idx):
        check_nan(val_batch, self)
        loss, batch_loss, mean_loss = self.compute_loss(val_batch)
        if mean_loss is None:
            return None
        else:
            self.log('val_loss', mean_loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
