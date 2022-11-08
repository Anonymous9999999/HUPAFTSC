import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from ..loss_functions import ContrastiveLoss
from ..loss_functions import TemporalContrastingLoss
from ...utils import check_nan
from ...utils import get_X_padding_masks
from ...ts_aug import ts_aug
from .pretrain_method_utils import TsGeneralPretrainDataset
from .pretrain_hp_config import PretrainHPConfig


class TsTccHP(PretrainHPConfig):
    def __init__(self, is_grid_search):

        default_hp = {'lambda1': 0.5,
                      'lambda2': 1.0,
                      'temporal_k_ratio': 0.4,
                      'temporal_max_k_sample': 32,
                      'contrastive_t': 0.5}
        hp_range = {'lambda1': [0.5, 1.0],
                    'lambda2': [0.5, 1.0],
                    'temporal_k_ratio': [0.2, 0.4],
                    'temporal_max_k_sample': [32, 64],
                    'contrastive_t': [0.5]}
        super(TsTccHP, self).__init__(default_hp, hp_range, is_grid_search=is_grid_search)


class TsTcc:
    def __init__(self):
        pass

    @staticmethod
    def collate_unsuperv(X):
        #  (batch_size, seq_len, feat_dimension)
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
                                  collate_fn=lambda x: TsTcc.collate_unsuperv(x))
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                collate_fn=lambda x: TsTcc.collate_unsuperv(x))
        return (train_loader, val_loader)


class TsTccPretrainWrapper(pl.LightningModule):
    """
    Time-Series Representation Learning via Temporal and Contextual Contrasting
    https://www.ijcai.org/proceedings/2021/0324.pdf
    """

    def __init__(self,
                 batch_size,
                 encoder,
                 input_dim,
                 project_dim=32,
                 is_use_temporal_loss=True,
                 lambda1=1.0,
                 lambda2=0.7,
                 temporal_k_ratio=0.2,
                 temporal_max_k_sample=32,
                 contrastive_t=0.5,
                 ):
        super(TsTccPretrainWrapper, self).__init__()
        self.encoder = encoder
        self.batch_size = batch_size
        self.pretrain_head = nn.Linear(encoder.output_dim, project_dim)
        self.input_project = nn.Linear(input_dim, project_dim)
        self.contrastive_loss = ContrastiveLoss(batch_size=batch_size,
                                                temperature=contrastive_t)
        self.temporal_loss = TemporalContrastingLoss(k_ratio=temporal_k_ratio,
                                                     max_k_sample=temporal_max_k_sample)
        self.is_use_temporal_loss = is_use_temporal_loss
        self.lambda1 = lambda1
        self.lambda2 = lambda2

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
        output = torch.mean(output, dim=1)
        return output

    def compute_loss(self, batch_data):
        X, padding_masks = batch_data
        device = X.device
        weak_aug_X, strong_aug_X = ts_aug(X, is_sample_seq_first=True, to_torch=True)
        weak_aug_X, strong_aug_X = weak_aug_X.to(device), strong_aug_X.to(device)
        self.contrastive_loss.batch_size = weak_aug_X.shape[0]
        weak_aug_embdding, strong_aug_embedding = self.forward(weak_aug_X), self.forward(strong_aug_X)
        contrastive_loss_value = self.contrastive_loss.forward(weak_aug_embdding, strong_aug_embedding)

        if self.is_use_temporal_loss:
            temporal_loss_value = self.temporal_loss.forward(self, weak_aug_X, strong_aug_X)
            loss = self.lambda1 * temporal_loss_value + self.lambda2 * contrastive_loss_value
        else:
            loss = contrastive_loss_value
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
