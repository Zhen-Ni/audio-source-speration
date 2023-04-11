#!/usr/bin/env python3

import torch
from .utils import ceil

__all__ = 'Demucs',



class Encoder(torch.nn.Module):
    """Encoder.

    Uses padding so that the output size is always ceil(input size / stride).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels * 2,
                                     kernel_size=1, stride=1)
        self.glu = torch.nn.GLU(dim=1)    # split input on dim 1 (channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outsize = ceil(x.shape[-1] / self.stride)
        insize = outsize * self.stride + self.kernel_size - self.stride
        pad = insize - x.shape[-1]
        padleft = pad // 2
        padright = pad - padleft
        x = torch.nn.functional.pad(x, (padleft, padright))
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.glu(x)
        return x


class Decoder(torch.nn.Module):
    """Decoder.

    The output can be trimed to given length.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, context: int,
                 final_layer: bool = False):
        super().__init__()
        self.is_final_layer = final_layer
        self._trim = context % 2 == 0
        padding = (context - 1) // 2 + self._trim
        self.conv = torch.nn.Conv1d(in_channels, in_channels * 2,
                                    kernel_size=context, stride=1,
                                    padding=padding)
        self.glu = torch.nn.GLU(dim=1)    # split input on dim 1 (channels)
        self.padding = (kernel_size - stride) // 2
        self.convt = torch.nn.ConvTranspose1d(in_channels,
                                              out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride)
        if not self.is_final_layer:
            self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor,
                length: int | None = None) -> torch.Tensor:
        x = self.conv(x)
        if self._trim:
            x = x[..., :-1]
        x = self.glu(x)
        x = self.convt(x)
        if length:
            trim = x.shape[-1] - length
            trimleft = trim // 2
            trimright = trim - trimleft
            x = x[..., trimleft: -trimright]
        if not self.is_final_layer:
            x = self.relu(x)
        return x


class BLSTM(torch.nn.Module):
    def __init__(self, input_size: int, num_layers: int):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size=input_size,
                                  num_layers=num_layers,
                                  bidirectional=True)
        self.linear = torch.nn.Linear(input_size * 2, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape of x now: (batch_size, nchannels, time_points)
        x = x.permute(2, 0, 1)
        # Shape of x now: (time_points, batch_size, nchannels)
        x = self.lstm(x)[0]                     # only output needed
        # Shape of x now: (time_points, batch_size, hidden_size*2)
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        return x


class Demucs(torch.nn.Module):
    """Demucs model to perform source seperation.

    Parameters
    ----------
    sources: int
        Number of sources to separate.
    channels: int
        Stereo or mono.
    depth: int
        Number of encoders/decoders .
    initial_growth: int
        Multiply number of channels for the first encoder.
    growth: int
        Multiply number of channels for the remaining layers of the encoder.
    lstm_layers: int
        Number of lstm layers, 0 = no lstm
    kernel_size: int
        Kernel size for convolution..
    stride: int
        Stride for convolution.
    context: int
        Context size for convolution of decoders.
    """

    def __init__(self,
                 sources: int,
                 channels: int,
                 depth: int = 6,
                 initial_growth: int = 32,
                 growth: int = 2,
                 lstm_layers: int = 2,
                 kernel_size: int = 8,
                 stride: int = 4,
                 context: int = 3):
        super().__init__()
        self.sources = sources
        self.channels = channels
        self.depth = depth

        self.encoders = torch.nn.ModuleList()
        self.decoders = torch.nn.ModuleList()

        in_channels = channels
        out_channels = in_channels * initial_growth
        for i in range(depth):
            self.encoders.append(Encoder(in_channels, out_channels,
                                         kernel_size, stride))
            if i == 0:
                in_channels *= sources
            final_layer = True if i == 0 else False
            self.decoders.insert(0,
                                 Decoder(out_channels, in_channels,
                                         kernel_size, stride, context,
                                         final_layer=final_layer))
            in_channels = out_channels
            out_channels *= growth
        if lstm_layers:
            self.lstm = BLSTM(in_channels, lstm_layers)
        else:
            self.lstm = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape of x now: (batch_size, nchannels, time_points)
        saved = [x]
        for encoder in self.encoders:
            x = encoder(x)
            saved.append(x)
        x = self.lstm(x)
        for decoder in self.decoders:
            skip = saved.pop()
            x = decoder(x + skip, length=saved[-1].shape[-1])
        x = x.view(x.size(0), self.sources, self.channels, x.size(-1))
        # Shape of x now: (batch_size, nsources (4), nchannels, time_points)
        return x


