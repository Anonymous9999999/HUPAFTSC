import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, bidirectional=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=0.0)

    def forward(self, batch_x, padding_masks):

        if padding_masks is None:
            lengths = [batch_x.shape[1] for _ in batch_x]
        else:
            lengths = []
            for mask in padding_masks:
                lengths.append(torch.where(mask == True)[0][-1].item() + 1)

        total_length = batch_x.shape[1]
        packed_batch_x = pack_padded_sequence(batch_x, batch_first=True, lengths=lengths, enforce_sorted=False)
        # The first item in the returned tuple of pack_padded_sequence is a data (tensor) --
        # a tensor containing the packed sequence. The second item is a tensor of integers holding information about
        # the batch size at each sequence step.
        # hn: (num_layers * 2, batch_size, hidden_dim), containing the hidden state for `t = seq_len`
        # cn: (num_layers * 2, batch_size, hidden_dim), containing the cell state for `t = seq_len`
        output, (hn, cn) = self.lstm(packed_batch_x)
        output = pad_packed_sequence(output, batch_first=True, total_length=total_length)
        output = output[0]
        return output

    @property
    def output_dim(self):
        return self.hidden_dim * 2

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
