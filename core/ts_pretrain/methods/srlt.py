import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from ..loss_functions import TripletLossVaryingLength
from ...utils import check_nan
from ...utils import get_X_padding_masks
from .pretrain_method_utils import TsGeneralPretrainDataset
from .pretrain_hp_config import PretrainHPConfig


class SRLTHP(PretrainHPConfig):
    def __init__(self, is_grid_search):
        default_hp = {'nb_random_samples': 3,
                      'negative_penalty': 2,
                      'min_len': 8
                      }
        hp_range = {'nb_random_samples': [3, 4, 5],
                    'negative_penalty': [1, 1.5, 2],
                    'min_len': [2, 4, 8]
                    }
        super(SRLTHP, self).__init__(default_hp, hp_range, is_grid_search=is_grid_search)


class SRLT:
    def __init__(self):
        pass

    @staticmethod
    def collate_unsuperv(X):
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
                                  collate_fn=lambda x: SRLT.collate_unsuperv(x))
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                collate_fn=lambda x: SRLT.collate_unsuperv(x))
        return (train_loader, val_loader)


class SrltTripletLossPretrainWrapper(pl.LightningModule):
    def __init__(self,
                 encoder,
                 data,
                 project_dim=32,
                 nb_random_samples=3,
                 negative_penalty=1,
                 min_len=2):
        super(SrltTripletLossPretrainWrapper, self).__init__()
        self.encoder = encoder
        data = torch.Tensor(data)
        self.data_mask, _ = get_X_padding_masks(data)
        self.data = self._fill_data_with_nan(data, self.data_mask)
        self.pretrain_head = nn.Linear(encoder.output_dim, project_dim)
        self.loss_func = TripletLossVaryingLength(compared_length=None,
                                                  nb_random_samples=nb_random_samples,
                                                  negative_penalty=negative_penalty,
                                                  min_len=min_len)

    def _fill_data_with_nan(self, data, data_mask):
        for x, mask in zip(data, data_mask):
            x[~mask] = torch.nan
        return data

    def forward(self, X, padding_masks=None, is_permute=False):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
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
        X = self._fill_data_with_nan(X, padding_masks)
        batch_loss = self.loss_func.forward(X,
                                            self,
                                            train=self.data,
                                            save_memory=False)
        loss = batch_loss
        mean_loss = loss / len(X)
        return loss, batch_loss, mean_loss

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
