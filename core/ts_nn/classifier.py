import torch
from torch import nn
import pytorch_lightning as pl
from ..utils import check_nan


class TSClassifier(pl.LightningModule):
    def __init__(self,
                 num_classes,
                 max_len,
                 encoder,
                 lr):
        super(TSClassifier, self).__init__()
        self.encoder = encoder
        self.use_mean_pool = True
        if self.use_mean_pool:
            self.conv = torch.nn.Conv1d(max_len, 1, 1)
            self.classifier_head = nn.Sequential(
                nn.Linear(encoder.output_dim, encoder.output_dim),
                nn.BatchNorm1d(encoder.output_dim),
                nn.ReLU(),
                nn.Dropout(p=0.02),
                nn.Linear(encoder.output_dim, encoder.output_dim),
                nn.BatchNorm1d(encoder.output_dim),
                nn.ReLU(),
                nn.Dropout(p=0.02),
                nn.Linear(encoder.output_dim, num_classes)
            )
        else:
            self.classifier_head = nn.Sequential(
                nn.Linear(encoder.output_dim * max_len, encoder.output_dim),
                nn.BatchNorm1d(encoder.output_dim),
                nn.ReLU(),
                nn.Dropout(p=0.02),
                nn.Linear(encoder.output_dim, encoder.output_dim),
                nn.BatchNorm1d(encoder.output_dim),
                nn.ReLU(),
                nn.Dropout(p=0.02),
                nn.Linear(encoder.output_dim, num_classes)
            )
        self.loss_func = nn.CrossEntropyLoss()
        self.lr = lr

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

        if self.use_mean_pool:
            output = self.conv(output).squeeze(1)
        else:
            output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)

        output = self.classifier_head(output)  # (batch_size, num_classes)
        return output

    def forward_softmax(self, batch_data):
        """
        Only used for test

        Parameters
        ----------
        batch_data

        Returns
        -------

        """
        X, _, X_padding_mask = batch_data
        X = X.to(self.device)
        X_padding_mask = X_padding_mask.to(self.device)
        output = self.forward(X, X_padding_mask)
        return nn.Softmax(dim=1)(output)

    def compute_loss(self, batch_data):
        X, actual_Y, X_padding_mask = batch_data
        predicted_Y = self.forward(X, X_padding_mask)
        loss = self.loss_func(predicted_Y, actual_Y)
        return loss

    def training_step(self, train_batch, batch_idx):
        check_nan([train_batch[0], train_batch[1]], self)
        mean_loss = self.compute_loss(train_batch)
        self.log('train_loss', mean_loss, prog_bar=True, on_step=False, on_epoch=True)
        return mean_loss

    def validation_step(self, val_batch, batch_idx):
        check_nan([val_batch[0], val_batch[1]], self)
        mean_loss = self.compute_loss(val_batch)
        self.log('val_loss', mean_loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, val_batch, batch_idx):
        check_nan([val_batch[0], val_batch[1]], self)
        mean_loss = self.compute_loss(val_batch)
        self.log('test_loss', mean_loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
