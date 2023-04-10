#!/usr/bin/env python3


import torch

from demucs import Demucs, Audioset, Trainer, Tester
from demucs.augment import Shift, FlipChannels, FlipSign, Scale, Remix


if __name__ == '__main__':
    epochs = 2
    model_name = 'demucs.th'
    augment = torch.nn.Sequential(
        Shift(), FlipChannels(), FlipSign(), Scale(), Remix())

    print('Start to load data.')

    trainset = Audioset("../musdb18/train/train",
                        samples=44100, stride=4410000)
    validset = Audioset("../musdb18/train/valid",
                        samples=44100, stride=4410000)
    testset = Audioset("../musdb18/test", samples=44100, stride=44100000000)

    print('Data loaded.\n')
    print('Start training process.')

    best_loss: float | None
    try:
        trainer = Trainer.load('trainer.trainer')
        best_loss = min(trainer.history['validate_loss'])
        print("Use stored trainer, continue training.")
    except FileNotFoundError:
        model = Demucs(4, 2, depth=2, initial_growth=2, lstm_layers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        critrion = torch.nn.L1Loss()
        trainer = Trainer(model, optimizer, critrion)
        best_loss = None
        print('New trainer generated.')
    print(f'current/total epochs [{trainer.epoch}/{epochs}]')
    while trainer.epoch < epochs:
        trainer.train(trainset, augment, batch_size=1)
        trainer.validate(validset, batch_size=1)
        trainer.save(device='cpu')
        # Save the best model.
        current_loss = trainer.history['validate_loss'][-1]
        if (best_loss is None) or (current_loss < best_loss):
            print('This model will be saved as the best model.')
            best_loss = current_loss
            with open(model_name, 'wb') as f:
                torch.save(trainer.model, f)

    print('Training process finished.\n')
    print('Start to test the best model.')

    with open(model_name, 'rb') as f:
        model = torch.load(f, "cpu")
    try:
        tester = Tester.load('tester.tester')
        print("Use stored tester, continue testing.")
    except FileNotFoundError:
        tester = Tester()
        print('New tester generated.')
    meters = tester.evaluate(model, testset)

    print('Test process finished.')
