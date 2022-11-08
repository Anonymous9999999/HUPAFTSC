from torch import nn
import torch.nn.functional as F


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, input_dims, in_channels, channels, kernel_size):
        super().__init__()
        self.input_fc = nn.Linear(input_dims, channels[0])
        self.channels = channels
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1)
            )
            for i in range(len(channels))
        ])
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x, padding_masks):
        projected_x = self.input_fc(x)
        projected_x = projected_x.permute(0, 2, 1)
        output = self.repr_dropout(self.net(projected_x))
        output = output.permute(0, 2, 1)
        return output

    @property
    def output_dim(self):
        return self.channels[-1]

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
