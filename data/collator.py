import torch
import numpy as np

from .preprocessing import *


def reprocess_tensor(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    durations = [batch[ind]["duration"] for ind in cut_list]
    energies = [batch[ind]["energy"] for ind in cut_list]
    pitches = [batch[ind]["pitch"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.size(0))

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(
            np.pad(
                [i+1 for i in range(int(length_src_row))],
                (0, max_len-int(length_src_row)),
                'constant',
            ),
        )
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.size(0))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(
            np.pad(
                [i+1 for i in range(int(length_mel_row))],
                (0, max_mel_len-int(length_mel_row)),
                'constant',
            ),
        )
    mel_pos = torch.from_numpy(np.array(mel_pos))

    texts = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations)
    mel_targets = pad_2D_tensor(mel_targets)
    energies = pad_1D_tensor(energies)
    pitches = pad_1D_tensor(pitches)

    out = {
        "text": texts,
        "mel_target": mel_targets,
        "duration": durations,
        "mel_pos": mel_pos,
        "src_pos": src_pos,
        "mel_max_len": max_mel_len,
        "pitch": pitches,
        "energy": energies,
    }

    return out


def collate_fn_tensor(batch, train_config):
    len_arr = np.array([d["text"].size(0) for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = batchsize // train_config.batch_expand_size

    assert real_batchsize != 0

    cut_list = list()
    for i in range(train_config.batch_expand_size):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(train_config.batch_expand_size):
        output.append(reprocess_tensor(batch, cut_list[i]))

    return output
