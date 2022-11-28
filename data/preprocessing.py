import os
import glob
import time

# import librosa
import torch
import torch.nn.functional as F
import numpy as np
import pyworld as pw

from tqdm import tqdm
from pathlib import Path
from text import text_to_sequence


def pad_1d(inputs, pad=0):
    def pad_data(x, length, pad):
        x_padded = np.pad(
            x,
            (0, length - x.shape[0]),
            mode='constant',
            constant_values=pad,
        )
        return x_padded
    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, pad) for x in inputs])
    return padded


def pad_1D_tensor(inputs, pad=0):
    def pad_data(x, length, pad):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded
    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, pad) for x in inputs])
    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        pad = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")
        s = np.shape(x)[1]
        x_padded = np.pad(
            x,
            (0, max_len - np.shape(x)[0]),
            mode='constant',
            constant_values=pad,
        )
        return x_padded[:, :s]
    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])
    return output


def pad_2D_tensor(inputs, maxlen=None):
    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")
        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len-x.size(0)))
        return x_padded[:, :s]
    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])
    return output


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)
    return txt


def get_data_to_buffer(train_config):
    '''text, alignments, and ground-truth mel specs'''
    buffer = list()

    text = process_text(train_config.data_path)
    wavs = process_wavs(train_config.wavs_path)

    pitches_path = Path(train_config.pitches_path)
    energies_path = Path(train_config.energies_path)

    for i in tqdm(range(len(text))):
        # mel specs
        mel_gt_name = os.path.join(
            train_config.mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1),
        )
        mel_gt_target = np.load(mel_gt_name)

        # duration
        duration = np.load(
            os.path.join(
                train_config.alignment_path, str(i)+".npy",
            )
        )

        # text
        character = text[i][0:len(text[i])-1]
        character = np.array(
            text_to_sequence(
                character, train_config.text_cleaners,
            ),
        )

        # pitch and energy 
        wav_name = wavs[i].split('/')[-1].split('.')[0] + '.pt'
        pitch = torch.load(os.path.join(pitches_path, wav_name))
        energy = torch.load(os.path.join(energies_path, wav_name))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)

        buffer.append({
            "text": character,  # torch
            "duration": duration,  # torch
            "mel_target": mel_gt_target,  # torch
            "pitch": pitch,  # torch
            "energy": energy,  # torch    
        })

    return buffer


def process_wavs(wavs_path):
    wavs = glob.glob(wavs_path + "/*.wav")
    wavs.sort()
    return wavs
