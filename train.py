#!/usr/bin/env python3

from __future__ import annotations
import sys
import time
import logging
import pickle

import tqdm
import torch


from utils import copy_to


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
WORKERS = 4


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    # formatter = logging.Formatter('%(asctime)s %(message)s')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    return logger


class AverageMeter:
    """Computes and stores the current and weighted average value."""
    __slots__ = 'average', 'value', 'sum', 'count'

    def __init__(self):
        self.average = 0.
        self.value = 0.
        self.sum = 0.
        self.count = 0

    def update(self, value: float, n: int = 1) -> AverageMeter:
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count
        return self


class Trainer():
    """Class for training a model."""

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 critrion: torch.nn.Module,
                 *,
                 device: torch.device | int | str | None = None,
                 start_epoch: int = 0,
                 filename: str | None = None,
                 milestones: list[int] = [100, 150],
                 gamma: float = 0.1,
                 ):
        self.model = model
        self.optimizer = optimizer
        self.critrion = critrion          # loss function
        # Use property setter to move model and loss function to target device
        self.device = DEVICE if device is None else device

        self.epoch = start_epoch
        self.filename = 'trainer.trainer' if filename is None else filename
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, gamma=gamma, milestones=milestones,
            last_epoch=self.epoch-1)
        self.history: dict[str, list[float]] = {'train_loss': [],
                                                'validate_loss': [],
                                                }

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device | int | str) -> Trainer:
        self._device = torch.device(device)
        self.model.to(self._device)
        self.critrion.to(self._device)
        return self

    @property
    def lr(self) -> list[float]:
        return [pg['lr'] for pg in self.optimizer.param_groups]

    @lr.setter
    def lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save(self, device: torch.device | int | str = "cpu"):
        data = copy_to(self.__dict__, torch.device(device))
        with open(self.filename, 'wb') as f:
            f.write(pickle.dumps((data, self.device)))

    def save_as(self, filename: str):
        self.filename = filename
        return self.save()

    @staticmethod
    def load(filename: str,
             device: torch.device | int | str | None = None
             ) -> Trainer:
        with open(filename, 'rb') as f:
            data, default_device = pickle.loads(f.read())
        if device is None:
            data = copy_to(data, default_device)
        else:
            data = copy_to(data, torch.device(device))
        res = object.__new__(Trainer)
        res.__dict__.update(data)
        return res

    def train(self,
              dataset: torch.utils.data.Dataset,
              augment: torch.nn.Module = None,
              batch_size: int = BATCH_SIZE,
              shuffle: bool = True,
              workers: int = WORKERS,
              ) -> AverageMeter:
        "Train the model by given dataloader."
        print(f'    ---- Epoch {self.epoch} ----    ')
        t_start = time.time()
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=workers)
        self.model.train()
        loss_meter = AverageMeter()
        tq = tqdm.tqdm(loader,
                       desc=f"train",
                       ncols=None,
                       leave=False,
                       file=sys.stdout,
                       unit=" batch")
        for streams in tq:
            if len(streams) < batch_size:
                # Skip incomplete batch for augment.Remix to work properly
                continue
            streams = streams.to(self.device)
            sources = streams[:, 1:]
            if augment is not None:
                augment = augment.to(self.device)
                sources = augment(sources)
            mix = torch.sum(sources, dim=1)

            estimates = self.model(mix)
            loss = self.critrion(estimates, sources)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), batch_size)
            tq.set_postfix(loss=f"{loss_meter.value:.4f}")

        print(f'train result: '
              f'avg loss = {loss_meter.average:.4f}, '
              f'wall time = {time.time()- t_start:.2f}s')
        self.scheduler.step()
        # Save information for this epoch.
        self.epoch += 1
        self.history['train_loss'].append(loss_meter.average)
        return loss_meter

    def validate(self,
                 dataset: torch.utils.data.Dataset,
                 batch_size: int = BATCH_SIZE,
                 shuffle: bool = True,
                 workers: int = WORKERS,
                 ) -> AverageMeter:
        t_start = time.time()
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=workers)
        self.model.eval()
        loss_meter = AverageMeter()
        tq = tqdm.tqdm(loader,
                       desc=f"valid",
                       ncols=None,
                       leave=False,
                       file=sys.stdout,
                       unit=" batch")
        for streams in tq:
            current_batch_size = streams.size(0)
            streams = streams.to(self.device)
            sources = streams[:, 1:]
            mix = torch.sum(sources, dim=1)

            with torch.no_grad():
                estimates = self.model(mix)
            loss = self.critrion(estimates, sources)
            loss_meter.update(loss.item(), current_batch_size)
            tq.set_postfix(loss=f"{loss_meter.value:.4f}")
        print(f'valid result: '
              f'avg loss = {loss_meter.average:.4f}, '
              f'wall time = {time.time()- t_start:.2f}s')
        # Save test results only the fisrt run.
        if len(self.history['validate_loss']) < self.epoch:
            self.history['validate_loss'].append(loss_meter.average)
        return loss_meter


if __name__ == '__main__':
    from demucs import Demucs
    from musdb import Audioset
    from augment import FlipChannels, FlipSign
    
    testset = Audioset("../musdb18/test", samples=44100, stride=441000000//2)
    augment = torch.nn.Sequential(FlipChannels(), FlipSign())

    model = Demucs(4, 2, depth=2, initial_growth=8, lstm_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    critrion = torch.nn.L1Loss()

    trainer = Trainer(model, optimizer, critrion)
    loss_meter = trainer.train(testset, augment, batch_size=1)
    loss_meter2 = trainer.validate(testset, batch_size=1)
