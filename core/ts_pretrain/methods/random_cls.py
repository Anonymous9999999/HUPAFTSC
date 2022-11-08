import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Callback
from ...utils import check_nan
from ...utils import get_X_padding_masks
from .pretrain_method_utils import TsGeneralPretrainDataset
from .pretrain_hp_config import PretrainHPConfig


class RandomClassCallback(Callback):
    def __init__(self):
        super().__init__()
        self.p_l2 = {}
        self.p_l2_norm = {}

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for name, params in pl_module.named_parameters():
            self.p_l2[name] = torch.mean(params.detach().cpu().abs()).item()
            self.p_l2_norm[name] = params.data.norm(2)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for name, params in pl_module.named_parameters():
            p_l2_start = self.p_l2[name]
            p_l2_end = torch.mean(params.detach().cpu().abs()).item()
            new_params = (p_l2_start / p_l2_end) * params.detach().cpu()
            params.data = new_params

class RandomClsHP(PretrainHPConfig):
    def __init__(self, is_grid_search):
        default_hp = {'cls_n': 8}
        hp_range = {'cls_n': [2, 4, 8, 16, 32]}
        super(RandomClsHP, self).__init__(default_hp, hp_range, is_grid_search=is_grid_search)


class RandomCls:
    def __init__(self):
        pass

    @staticmethod
    def collate_unsuperv(X, cls_n):
        X = torch.stack(X).float()
        Y = torch.randint(0, cls_n, (X.shape[0], 1))
        padding_masks, lengths = get_X_padding_masks(X)
        return X, Y, padding_masks

    @staticmethod
    def pretrain_dataloader(train_X, val_X, batch_size, cls_n):
        train_dataset = TsGeneralPretrainDataset(train_X)
        val_dataset = TsGeneralPretrainDataset(val_X)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True,
                                  collate_fn=lambda x: RandomCls.collate_unsuperv(x, cls_n))
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                collate_fn=lambda x: RandomCls.collate_unsuperv(x, cls_n))
        return (train_loader, val_loader)


class RandomClsLossPretrainWrapper(pl.LightningModule):
    def __init__(self,
                 encoder,
                 max_seq_len,
                 cls_n,
                 hp=None):
        super(RandomClsLossPretrainWrapper, self).__init__()
        self.encoder = encoder
        self.pretrain_head = nn.Linear(encoder.output_dim * max_seq_len, cls_n)
        self.loss_func = nn.CrossEntropyLoss()
        self.hp = hp

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        if padding_masks is None:
            padding_masks = torch.ones((X.shape[0], X.shape[1]), dtype=torch.bool).to(X.device)
        output = self.encoder.forward(X, padding_masks)
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings

        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.pretrain_head(output)  # (batch_size, num_classes)
        return output

    def compute_loss(self, batch_data):
        X, actual_Y, X_padding_mask = batch_data
        predicted_Y = self.forward(X, X_padding_mask)
        loss = self.loss_func(predicted_Y, actual_Y.flatten())
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
