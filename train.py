import os
import logging
import argparse
from datetime import datetime
import typing as tp

import torch 
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from configs import (
    TrainConfig,
    FastSpeechConfig,
    MelSpectrogramConfig,
)
from data import get_dataloader
from model import FastSpeech2, FastSpeechLoss2
from wandb_writer import WanDBWriter
from pathlib import Path

Loader = tp.TypeVar('Loader')
Model = tp.TypeVar('Model')
Loss = tp.TypeVar('Loss')
Optim = tp.TypeVar('Optim')
Schdlr = tp.TypeVar('Schdlr')
Logger = tp.TypeVar('Logger')

logging.basicConfig(level=logging.INFO)


def now():
    return datetime.now().strftime('%Y-%m-%d_%H-%M')


def train(
    train_config: TrainConfig,
    training_loader: Loader,
    model: Model,
    fs2_loss: Loss,
    optimizer: Optim,
    scheduler: Schdlr,
    logger: Logger,
    args: dict,
) -> None:
    current_step = 0

    epochs = train_config.epochs
    expand_size = train_config.batch_expand_size
    num_batches = len(training_loader)

    tqdm_bar = tqdm(
        total=epochs * num_batches * expand_size - current_step
    )

    for epoch in range(epochs):
        for i, batchs in enumerate(training_loader):
            for j, db in enumerate(batchs):
                current_step += 1
                tqdm_bar.update(1)

                logger.set_step(current_step)

                # Get Data
                src_seq = db["text"].long().to(train_config.device)
                mel_target = db["mel_target"].float().to(train_config.device)
                duration = db["duration"].int().to(train_config.device)
                mel_pos = db["mel_pos"].long().to(train_config.device)
                src_pos = db["src_pos"].long().to(train_config.device)
                pitch = db["pitch"].float().to(train_config.device)
                energy = db["energy"].float().to(train_config.device)
                max_mel_len = db["mel_max_len"]

                # Forward
                (
                    output, 
                    log_duration_prediction,
                    pitch_prediction,
                    energy_prediction,
                ) = model(
                    src_seq=src_seq, 
                    src_pos=src_pos,
                    mel_pos=mel_pos,
                    mel_max_length=max_mel_len,
                    length_target=duration,
                    pitch_target=pitch,
                    energy_target=energy,
                )

                # Calc Loss
                (
                    total_loss,
                    mel_loss, 
                    duration_loss, 
                    pitch_loss, 
                    energy_loss,
                ) = fs2_loss(
                    mel_target=mel_target,
                    mel_predicted=output,
                    mel_pos=mel_pos,
                    duration_target=duration,
                    log_duration_predicted=log_duration_prediction,
                    pitch_target=pitch,
                    pitch_predicted=pitch_prediction,
                    energy_target=energy,
                    energy_predicted=energy_prediction,
                    src_pos=src_pos,
                )

                # Backward
                total_loss.backward()

                # Logger
                if current_step % train_config.log_step == 0:
                    logger.add_scalar("total_loss", total_loss.detach().cpu().numpy())
                    logger.add_scalar("mel_loss", mel_loss.detach().cpu().numpy())
                    logger.add_scalar("duration_loss", duration_loss.detach().cpu().numpy())
                    logger.add_scalar("pitch_loss", pitch_loss.detach().cpu().numpy())
                    logger.add_scalar("energy_loss", energy_loss.detach().cpu().numpy())

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    train_config.grad_clip_thresh,
                )

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                if current_step % train_config.save_step == 0:
                    save_path = os.path.join(
                        train_config.checkpoint_path, f'{args.project}/{args.version}/{now()}.pth'
                    )
                    torch.save(
                        {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, 
                        save_path,
                    )
                    logging.critical(f'save checkpoint in {save_path}')


def main():
    args = parse_args()
    
    train_config = TrainConfig()
    setattr(train_config, 'wandb_project', args.project)
    setattr(train_config, 'wandb_version', args.version)
    if args.batch_size is not None:
        setattr(train_config, 'batch_size', args.batch_size)
    setattr(train_config, 'device', torch.device('cuda:' + args.device))

    logging.critical(f' {train_config.device} will be used! ')
    
    model_config = FastSpeechConfig()
    mel_config = MelSpectrogramConfig()

    # set logger
    logger = WanDBWriter(train_config)

    # set loader
    training_loader = get_dataloader(train_config)

    # set loss and model
    model = FastSpeech2(model_config, mel_config)
    model = model.to(train_config.device)
    fs2_loss = FastSpeechLoss2()

    # set optim
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # set scheduler
    steps_per_epoch = len(training_loader) * train_config.batch_expand_size
    scheduler = OneCycleLR(
        optimizer=optimizer,
        steps_per_epoch=steps_per_epoch,
        epochs=train_config.epochs,
        anneal_strategy="cos",
        max_lr=train_config.learning_rate,
        pct_start=0.1,
    )

    # prepare checkpoint folder
    path = Path(train_config.checkpoint_path) / Path(f'{args.project}/{args.version}')
    if not path.exists(): 
        path.mkdir(parents=True, exist_ok=True)

    train(
        train_config,
        training_loader,
        model,
        fs2_loss,
        optimizer,
        scheduler,
        logger,
        args,
    )


def parse_args() -> None:
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument(
        '-d',
        '--device',
        type=str, 
        default="0",
        help='cuda device index',
    )
    parser.add_argument(
        '-v',
        '--version',
        type=str, 
        default=f"{now()}",
        help='wandb experiment version',
    )
    parser.add_argument(
        '-p',
        '--project',
        type=str, 
        default=f"fast_speech_2",
        help='wandb project',
    )
    parser.add_argument(
        '-bs',
        '--batch-size',
        type=int,
        default=None,
        help='batch size',
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
