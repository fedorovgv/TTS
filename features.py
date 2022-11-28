import os
import glob
import torch
import numpy as np
import librosa
import pyworld as pw
from tqdm import tqdm
from pathlib import Path
import typing as tp

from configs import TrainConfig, FastSpeechConfig
from sklearn.preprocessing import StandardScaler


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)
    return txt


def process_wavs(wavs_path):
    wavs = glob.glob(wavs_path + "/*.wav")
    wavs.sort()
    return wavs


def get_pitch(wav, duration, train_config):
    pitch, t = pw.dio(
        wav.astype(np.float64),
        train_config.sample_rate,
        frame_period=train_config.hop_length/train_config.sample_rate*1000,
    )
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, train_config.sample_rate)
    pitch = pitch[: sum(duration)]
    return pitch


def stft(train_config):
    f = lambda x: torch.stft(
        torch.tensor(x),
        n_fft=train_config.win_length, 
        hop_length=train_config.hop_length,
        normalized=train_config.normalized,
    )
    return f


@torch.jit.script
def get_energy(mel):
    mel = torch.norm(mel, p=2, dim=-1)
    mel = torch.norm(mel, p=2, dim=0)
    return mel


def preprocces(train_config):
    text = process_text(train_config.data_path)
    wavs = process_wavs(train_config.wavs_path)
    
    pitches_path = Path(train_config.pitches_path_unnorm)
    if not pitches_path.exists(): 
        pitches_path.mkdir(parents=True, exist_ok=True)
    
    energies_path = Path(train_config.energies_path_unnorm)
    if not energies_path.exists(): 
        energies_path.mkdir(parents=True, exist_ok=True)
    
    _stft = stft(train_config)
    
    for i in tqdm(range(len(text))):
        duration = np.load(
            os.path.join(
                train_config.alignment_path, str(i)+".npy",
            )
        )
        wav_name = wavs[i].split('/')[-1].split('.')[0] + '.pt'
        
        wav, sr = librosa.load(wavs[i])
        pitch = get_pitch(wav, duration, train_config)
        pitch = torch.from_numpy(pitch)
        torch.save(
            pitch,
            os.path.join(pitches_path, wav_name)
        )
        
        mel = _stft(wav)
        energy = get_energy(mel)
        torch.save(
            energy,
            os.path.join(energies_path, wav_name)
        )
    
    return None


def normalize_and_stats(
    path: str, 
    mean: tp.Optional[float] = None, 
    var: tp.Optional[float] = None,
) -> None:
    items = glob.glob(path + "/*.pt")
    
    if mean is None:
        scaler = StandardScaler()
        for item_path in tqdm(items):
            item = torch.load(item_path)
            scaler.partial_fit(
                item.reshape((-1, 1))
            )
        mean, var = scaler.mean_, scaler.var_
    
    path = Path(os.path.join(path + '_norm'))
    if not path.exists(): 
        path.mkdir(parents=True, exist_ok=True)
    print(f'normalized items will be saved in {path}')
    
    print(f'mean {mean} var {var}')

    _min, _max = 10e6, -10e6
    for i, item_path in tqdm(enumerate(items)):
        item = torch.load(item_path)
        item = (item - mean) / var
        torch.save(
            item,
            os.path.join(path, items[i].split('/')[-1])
        )
        _min = min(_min, item.min())
        _max = max(_max, item.max())
    
    print(f'min {_min} _max {_max}')
    return None


if __name__ == '__main__':
    train_comfig = TrainConfig()
    model_config = FastSpeechConfig()

    preprocces(train_comfig)

    if not hasattr(model_config, 'pitch_mean'):
        normalize_and_stats(train_comfig.pitches_path_unnorm)
        normalize_and_stats(train_comfig.energies_path_unnorm)
    else:
        normalize_and_stats(
            train_comfig.pitches_path_unnorm,
            model_config.pitch_mean,
            model_config.pitch_var,
        )
        normalize_and_stats(
            train_comfig.energies_path_unnorm,
            model_config.energy_mean,
            model_config.energy_var,
        )
