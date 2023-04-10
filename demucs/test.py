#!/usr/bin/env python3

from __future__ import annotations
import typing
import time
import sys
import pickle

from concurrent import futures

import tqdm
# Require mir_eval to perform signal-to-distortion ratio (SDR) calculation
from mir_eval import separation
import numpy as np
import torch

from .utils import AverageMeter

__all__ = 'Tester',

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
WORKERS = 2


class Tester:
    def __init__(self, *,
                 start_index: int = 0,
                 filename: str | None = None):
        self.index = start_index
        self.filename = 'tester.tester' if filename is None else filename
        self.history: dict[str, list[float]] = {'drums': [],
                                                'bass': [],
                                                'other': [],
                                                'vocals': []}

    def save(self):
        with open(self.filename, 'wb') as f:
            f.write(pickle.dumps(self))

    def save_as(self, filename: str):
        self.filename = filename
        return self.save()

    @ staticmethod
    def load(filename: str) -> Tester:
        with open(filename, 'rb') as f:
            tester = pickle.loads(f.read())
        return tester

    def average_SDR(self) -> float:
        sdrs = []
        for value in self.history.values():
            sdrs.append(np.mean(value))
        return np.mean(sdrs)

    def evaluate(self,
                 model: torch.nn.Module,
                 dataset: torch.utils.data.Dataset,
                 workers: int = WORKERS,
                 device: torch.device | int | str = "cpu",
                 save: bool = True
                 ) -> float:
        t_start = time.time()
        print(f'current/total samples [{self.index}/{len(dataset)}]')
        model.eval()
        tq = tqdm.tqdm(range(self.index, len(dataset)), desc="test",
                       leave=False, file=sys.stdout, unit=" stream",
                       initial=self.index)
        for i in tq:
            stream = dataset[i]
            streams = stream.unsqueeze(0).to(device)
            sources = streams[:, 1:]
            nstreams = sources.size(1)    # should be 4 for musdb
            mix = streams[:, 0]
            with torch.no_grad():
                estimates = model(mix.to(device))
            sdr = []       # SDRs of [drums, bass, other, vocals]
            for j in range(nstreams):
                sdrj = separation.bss_eval_sources(sources[0, j].numpy(),
                                                   estimates[0, j].numpy())[0].mean()
                sdr.append(sdrj)

            tq.set_postfix({'avg SDR': f"{np.mean(sdr):.4f}"})
            self.index += 1
            self.history['drums'].append(sdr[0])
            self.history['bass'].append(sdr[1])
            self.history['other'].append(sdr[2])
            self.history['vocals'].append(sdr[3])
            if save:
                self.save()
        avgsdr = self.average_SDR()
        print(f'test result: '
              f'avg sdr = {avgsdr:.4f}, '
              f'wall time = {time.time()- t_start:.2f}s')
        return avgsdr
