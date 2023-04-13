#!/usr/bin/env python3

from __future__ import annotations
import os
import subprocess as sp

import numpy as np
import torch
from torch.utils.data import Dataset

from .audiofile import AudioFile, audiofile
from .utils import ensure_dir


__all__ = 'AudioSet',


class AudioSet(torch.utils.data.Dataset):
    """Build dataset from audios in specified folder.

    Parameters
    ----------
    path: str
        Create set from audios files from given path.
    samples: int, optional
        Length of each item in dataset. If None or not given,
        the whole audio file is gathered as one sample.
        (Default to None)
    stride: int, optional
        Stride for gathering data, in frames. If None or not given,
        waveforms are extracted without overlap.
    streams: int, optional
        Number of streams to extract, must be spcified if raw files
        are used.
    samplerate: int, optional
        Specifify the samplerate of audiofile, useful only for
        compressed audio files. (default to None)
    """

    def __init__(self,
                 path: str,
                 samples: int | None = None,
                 stride: int | None = None,
                 streams: int | None = None,
                 channels: int | None = None,
                 samplerate: int | None = None
                 ):
        self._path = path
        self._samples = samples
        self._stride = samples if stride is None else stride
        self._streams = streams
        self._channels = channels
        self._samplerate = samplerate
        self._filelist = self._gather_audio_files()

        if self._samples is None:
            self._cumulative_sizes = np.arange(len(self.filelist)) + 1
        else:
            sizes = []
            for audio in self._filelist:
                size = (audio.frames() - self._samples) // self._stride + 1
                sizes.append(size)
            self._cumulative_sizes = np.cumsum(sizes)

    def _gather_audio_files(self) -> list[AudioFile]:
        filelist = []
        for f in sorted(os.listdir(self._path)):
            if afile := audiofile(os.path.join(self._path, f),
                                  self._streams, self._channels,
                                  self._samplerate):
                filelist.append(afile)
        return filelist

    def __len__(self):
        return self._cumulative_sizes[-1]

    def __getitem__(self, index) -> torch.Tensor:
        file_index = np.searchsorted(self._cumulative_sizes, index,
                                     side='right')
        local_index = (index - self._cumulative_sizes[file_index - 1] if
                       file_index else index)
        audio = self.filelist[file_index]
        offset = local_index * self._stride if self._stride else None
        wav = audio.read(offset, self._samples)
        return wav

    @property
    def filelist(self) -> list[AudioFile]:
        return self._filelist
