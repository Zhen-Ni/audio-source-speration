#!/usr/bin/env python3

from __future__ import annotations
import time
import pickle
from concurrent import futures
import typing

import tqdm
# Require mir_eval to perform signal-to-distortion ratio (SDR) calculation
from mir_eval import separation
import numpy as np
import torch

from .utils import copy_to, apply_model, free_memory

__all__ = 'Tester',

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
WORKERS = 2
SAMPLERATE = 44100
SEGMENT_FRAMES = SAMPLERATE * 10
OVERLAP_FRAMES = SAMPLERATE
NSHIFTS = 10
MAX_SHIFT = SAMPLERATE // 2


def bss_eval_sources(reference_sources: torch.Tensor,
                     estimated_sources: torch.Tensor,
                     compute_permutation: bool = True):
    # This function is the bottleneck of testing.
    # Shape of sources: nsources, nchannels, ntime
    nstreams = reference_sources.size(0)
    reference_sources = reference_sources.numpy()
    estimated_sources = estimated_sources.numpy()
    sdrs = []
    for i in range(nstreams):
        sdr, *_ = separation.bss_eval_sources(reference_sources[i],
                                              estimated_sources[i],
                                              compute_permutation)
        sdrs.append(np.mean(sdr))
    return sdrs


class Tester:
    def __init__(self, *,
                 device: torch.device | int | str | None = None,
                 start_index: int = 0,
                 segment_frames: int = SEGMENT_FRAMES,
                 overlap_frames: int = OVERLAP_FRAMES,
                 nshifts: int = NSHIFTS,
                 max_shift: int = MAX_SHIFT,
                 filename: str | None = None,
                 forced_gc: bool = False
                 ):
        # Use property setter to move model and loss function to target device
        self.device = DEVICE if device is None else device
        self.index = start_index
        self.segment_frames = segment_frames
        self.overlap_frames = overlap_frames
        self.nshifts = nshifts
        self.max_shift = max_shift
        self.filename = 'tester.tester' if filename is None else filename
        self.is_forced_gc = forced_gc
        self.history: dict[str, list[float]] = {'drums': [],
                                                'bass': [],
                                                'other': [],
                                                'vocals': []}

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device | int | str) -> Tester:
        self._device = torch.device(device)
        return self

    def save(self, device: torch.device | int | str = "cpu"):
        data = copy_to(self.__dict__, torch.device(device))
        with open(self.filename, 'wb') as f:
            f.write(pickle.dumps((self, self.device)))

    def save_as(self, filename: str):
        self.filename = filename
        return self.save()

    @ staticmethod
    def load(filename: str,
             device: torch.device | int | str | None = None
             ) -> Tester:
        with open(filename, 'rb') as f:
            tester, default_device = pickle.loads(f.read())
        if device is None:
            tester = copy_to(tester, default_device)
        else:
            tester = copy_to(tester, torch.device(device))
        return tester

    def average_SDR(self) -> float:
        sdrs = []
        for value in self.history.values():
            sdrs.append(np.mean(value))
        return np.mean(sdrs)

    def evaluate(self,
                 model: torch.nn.Module,
                 dataset: torch.utils.data.Dataset,
                 batch_size: int = BATCH_SIZE,
                 workers: int = WORKERS,
                 save: bool = True
                 ) -> float:
        t_start = time.time()
        print(f'current/total samples [{self.index}/{len(dataset)}]')
        subset = torch.utils.data.Subset(dataset,
                                         list(range(self.index, len(dataset))))
        loader = torch.utils.data.DataLoader(subset,
                                             batch_size=batch_size,
                                             num_workers=workers)
        model.eval()
        model.to(self.device)

        tq = tqdm.trange(len(subset),
                         desc='test',
                         ncols=None,
                         leave=False,
                         initial=self.index,
                         total=len(dataset),
                         unit='track'
                         )
        tqit = iter(tq)
        pendings = []
        with futures.ProcessPoolExecutor(workers) as pool:
            for streams in loader:
                current_batch_size = streams.size(0)
                streams = streams.to(self.device)
                sources = streams[:, 1:]
                mix = streams[:, 0]
                estimates = apply_model(model, mix,
                                        self.segment_frames,
                                        self.overlap_frames,
                                        self.nshifts,
                                        self.max_shift)
                sources = sources.cpu()
                estimates = estimates.cpu()
                for j in range(current_batch_size):
                    p = pool.submit(bss_eval_sources, sources[j], estimates[j])
                    pendings.append(p)

                del streams, sources, mix, estimates
                if self.is_forced_gc:
                    free_memory()

                # Procced to next outermost loop only when:
                # 1. no finished futures can be fetched (results must
                #    be fetched in sequence);
                # 2. number of pending futures is less than number of
                #    workers
                while True:
                    if pendings and pendings[0].done():
                        self._evaluate_gather_result(pendings.pop(0),
                                                     save, tq, tqit)
                    elif 2 * workers <= len(pendings):
                        time.sleep(0.1)
                    else:
                        break

            for p in pendings:
                self._evaluate_gather_result(p, save, tq, tqit)

        avgsdr = self.average_SDR()
        print(f'test result: '
              f'avg sdr = {avgsdr:.4f}, '
              f'wall time = {time.time()- t_start:.2f}s')
        return avgsdr

    def _evaluate_gather_result(self,
                                p: futures.Future,
                                save: bool,
                                tq: tqdm.tqdm,
                                tqit: typing.Iterator[tqdm.tqdm]
                                ):
        sdrs = p.result()
        self.index += 1
        self.history['drums'].append(sdrs[0])
        self.history['bass'].append(sdrs[1])
        self.history['other'].append(sdrs[2])
        self.history['vocals'].append(sdrs[3])
        if save:
            self.save()
        tq.set_postfix(sdr=f'{np.mean(sdrs):.4f}')
        next(tqit)
