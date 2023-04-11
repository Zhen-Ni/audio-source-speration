#!/usr/bin/env python3


import torch

from demucs import Demucs, Audioset, Trainer, Tester
from demucs.augment import Shift, FlipChannels, FlipSign, Scale, Remix


if __name__ == '__main__':
    epochs = 200
    sample_rate = 44100
    train_batch_size = 64
    lr = 3e-4
    eval_batch_size = 1
    eval_segment_frames = sample_rate * 600
    eval_overlap_frames = sample_rate * 1
    eval_nshifts = 10
    eval_max_shift = sample_rate // 2
    
    model_name = 'demucs.th'
    augment = torch.nn.Sequential(
        Shift(sample_rate),
        FlipChannels(),
        FlipSign(),
        Scale(min=0.25, max=1.25),
        Remix())

    print('Start loading data.')

    trainset = Audioset("../musdb18/train/train",
                        samples=sample_rate * 11, stride=sample_rate)
    validset = Audioset("../musdb18/train/valid")
    testset = Audioset("../musdb18/test")

    print('Data loaded.\n')
    print('Start training process.')

    best_loss: float | None
    try:
        trainer = Trainer.load('trainer.trainer')
        best_loss = min(trainer.history['validate_loss'])
        print("Use stored trainer, continue training.")
    except FileNotFoundError:
        model = Demucs(4, 2, depth=6, initial_growth=32, lstm_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        critrion = torch.nn.L1Loss()
        trainer = Trainer(model, optimizer, critrion, gamma=1.)
        best_loss = None
        print('New trainer generated.')
    print(f'current/total epochs [{trainer.epoch}/{epochs}]')
    while trainer.epoch < epochs:
        trainer.train(trainset, augment, batch_size=train_batch_size)
        trainer.validate(validset, batch_size=eval_batch_size,
                         segment_frames=eval_segment_frames,
                         overlap_frames=eval_overlap_frames,
                         nshifts=eval_nshifts,
                         max_shift=eval_max_shift)
        trainer.save(device='cpu')
        # Save the best model.
        current_loss = trainer.history['validate_loss'][-1]
        if (best_loss is None) or (current_loss < best_loss):
            print('This model will be saved as the best model.')
            best_loss = current_loss
            with open(model_name, 'wb') as f:
                torch.save(trainer.model, f)

    print('Training process finished.\n')
    print('Start testing the best model.')

    with open(model_name, 'rb') as f:
        model = torch.load(f, "cpu")
    try:
        tester = Tester.load('tester.tester')
        print("Use stored tester, continue testing.")
    except FileNotFoundError:
        tester = Tester(segment_frames=eval_segment_frames,
                        overlap_frames=eval_segment_frames,
                        nshifts=eval_nshifts,
                        max_shift=eval_max_shift)
        print('New tester generated.')
    meters = tester.evaluate(model, testset)

    print('Test process finished.')
