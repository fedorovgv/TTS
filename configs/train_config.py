import torch
from dataclasses import dataclass


@dataclass
class TrainConfig:
    checkpoint_path = "./model_ckpt"
    logger_path = "./logger"
    mel_ground_truth = "./datasets/mels"
    alignment_path = "./datasets/alignments"
    data_path = './datasets/train.txt'
    
    wandb_project = 'fastspeech'
    wandb_version = 'default'
    text_cleaners = ['english_cleaners']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 16
    epochs = 2000
    n_warm_up_step = 4000

    learning_rate = 1e-3
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    decay_step = [500000, 1000000, 2000000]
    
    save_step = 1000
    log_step = 5
    clear_Time = 20
    
    batch_expand_size = 32
    
    sample_rate: int = 22050
    wavs_path: str = './datasets/LJSpeech/wavs'
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    normalized: bool = True

    pitches_path_unnorm: str = './datasets/pitches'
    energies_path_unnorm: str = './datasets/energies'
    
    pitches_path: str = './datasets/pitches_norm'
    energies_path: str = './datasets/energies_norm'
