import os
import argparse
import logging
import typing as tp

import torch
import waveglow
import text
import audio
import utils

import numpy as np
from configs import (
    TrainConfig, FastSpeechConfig, MelSpectrogramConfig
)
from model import FastSpeech2

import warnings
warnings.filterwarnings('ignore')

Model = tp.TypeVar('Model')


def synthesis(
    model: Model, 
    phn: list, 
    train_config: TrainConfig, 
    alpha: float = 1.0, 
    beta: float = 1.0, 
    gamma: float = 1.0,
) -> tp.Tuple[torch.Tensor, ...]:
    
    text = np.array(phn)
    text = np.stack([text])

    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    
    sequence = torch.from_numpy(text).long().to(train_config.device)
    src_pos = torch.from_numpy(src_pos).long().to(train_config.device)

    with torch.no_grad():
        mel, *_ = model.forward(sequence, src_pos, alpha=alpha, beta=beta, gamma=gamma)
    
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def get_data(train_config: TrainConfig) -> list:
    tests = [
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
    ]
    data_list = list(text.text_to_sequence(test, train_config.text_cleaners) for test in tests)
    return data_list


def main():
    args = parse_args()

    train_config = TrainConfig()
    model_config = FastSpeechConfig()
    mel_config = MelSpectrogramConfig()

    data_list = get_data(train_config)

    WaveGlow = utils.get_WaveGlow()
    WaveGlow = WaveGlow.cuda()

    model = FastSpeech2(model_config, mel_config)
    model.load_state_dict(
        torch.load(args.model_path, map_location='cuda')['model']
    )
    model.cuda()
    model = model.eval()

    alpha = args.alpha
    beta = args.beta 
    gamma = args.gamma
    
    for i, phn in enumerate(data_list):
        mel, mel_cuda = synthesis(model, phn, train_config, alpha, beta, gamma)
        os.makedirs("results", exist_ok=True)
        audio.tools.inv_mel_spec(
            mel, f"results/s={alpha}_p={beta}_e={gamma}_{i}.wav"
        )
        waveglow.inference.inference(
            mel_cuda, WaveGlow,
            f"results/s={alpha}_p={beta}_e={gamma}_{i}.wav"
        )
        print(f'results/s={alpha}_p={beta}_e={gamma}_{i}.wav')


def parse_args() -> None:
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument(
        '-a',
        '--alpha',
        type=float, 
        default=1.0,
        help='speed control parameter',
    )
    parser.add_argument(
        '-b',
        '--beta',
        type=float, 
        default=1.0,
        help='pitch control parameter',
    )
    parser.add_argument(
        '-g',
        '--gamma',
        type=float, 
        default=1.0,
        help='energy control parameter',
    )
    parser.add_argument(
        '-p',
        '--model-path',
        type=str, 
        required=True,
        help='wandb experiment version',
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
