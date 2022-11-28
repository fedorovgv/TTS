import logging
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from configs import FastSpeechConfig
from data.preprocessing import pad_2D_tensor

logging.basicConfig(level=logging.DEBUG)


class VarianceAdaptor(nn.Module):
    def __init__(self, model_config: FastSpeechConfig):
        """ Variance Adaptor 
        Need to use nn.Parameter in pitch and energy bins because weights need to be loaded for inference.
        Linear quantization is used instead of logarithmic due to standard pitch and energy normalization.
        """
        super().__init__()

        self.duration_predictor = VariancePredictor(
            model_config,
        )
        self.length_regulator = LengthRegulator()

        self.pitch_predictor = VariancePredictor(
            model_config,
        )
        self.pitch_bins = nn.Parameter(
            torch.linspace(
                model_config.pitch_min, model_config.pitch_max, model_config.n_bins - 1,
            ),
            requires_grad=False,
        )
        self.pitch_embedding = nn.Embedding(
            model_config.n_bins, model_config.encoder_dim,
        )

        
        self.energy_predictor = VariancePredictor(
            model_config,
        )
        self.energy_bins = nn.Parameter(
            torch.linspace(
                model_config.energy_min, model_config.energy_max, model_config.n_bins - 1,
            ),
            requires_grad=False,
        )
        self.energy_embedding = nn.Embedding(
            model_config.n_bins, model_config.encoder_dim
        )


    def forward(
        self, 
        x: torch.Tensor, 
        duration_target: tp.Optional[torch.Tensor] = None, 
        max_len: tp.Optional[torch.Tensor] = None, 
        pitch_target: tp.Optional[torch.Tensor] = None, 
        energy_target: tp.Optional[torch.Tensor] = None, 
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ) -> tp.Tuple[torch.Tensor, ...]:
        
        log_duration_prediction = self.duration_predictor(x)

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
        else:
            duration_predictor_output = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * alpha), min=0,
            )
            if duration_predictor_output.dim() == 1:  # TODO: fix this for inference
                duration_predictor_output = duration_predictor_output.unsqueeze(0)
            x, mel_len = self.length_regulator(x, duration_predictor_output)

        pitch_prediction = self.pitch_predictor(x)
        if pitch_target is not None:
            pitch_embedding = self.pitch_embedding(
                torch.bucketize(pitch_target, self.pitch_bins)
            )
        else:
            pitch_prediction = pitch_prediction * beta
            pitch_embedding = self.pitch_embedding(
                torch.bucketize(pitch_prediction.detach(), self.pitch_bins)
            )
        x = x + pitch_embedding
        
        energy_prediction = self.energy_predictor(x)
        if energy_target is not None:
            energy_embedding = self.energy_embedding(
                torch.bucketize(energy_target, self.energy_bins)
            )
        else:
            energy_prediction = energy_prediction * gamma
            energy_embedding = self.energy_embedding(
                torch.bucketize(energy_prediction.detach(), self.energy_bins)
            )
        x = x + energy_embedding
        
        return (
            x, 
            log_duration_prediction, 
            pitch_prediction, 
            energy_prediction, 
            mel_len,
        )


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class VariancePredictor(nn.Module):
    """ Duration Predictor """
    def __init__(self, model_config: FastSpeechConfig):
        super().__init__()

        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.variance_predictor_filter_size
        self.kernel = model_config.variance_predictor_kernel_size
        self.conv_output_size = model_config.variance_predictor_filter_size
        self.dropout = model_config.dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)
        out = self.linear_layer(encoder_output)
        out = out.squeeze()
        return out


class LengthRegulator(nn.Module):
    """ 
    Length Regulator inspired from from https://github.com/ming024/FastSpeech2
    without matrix multiplication
    """
    def __init__(self) -> None:
        super().__init__()
    
    @staticmethod
    def LR(x, duration, max_len=None):
        
        def _expand(item, item_durations):
            out = list()
            for i, frame in enumerate(item):
                expanded_size = max(int(item_durations[i].item()), 0)
                out.append(frame.expand(expanded_size, -1))
            return torch.cat(out, 0), int(item_durations.sum().item())

        expanded_batch = list()
        mel_len = list()

        for item, item_durations in zip(x, duration):
            expanded, expanded_size = _expand(item, item_durations)
            expanded_batch.append(expanded)
            mel_len.append(expanded_size)
        
        expanded_batch = pad_2D_tensor(expanded_batch, max_len)
        
        return expanded_batch, torch.tensor(mel_len).long().to(x.device)
    
    def forward(self, x, duration, max_len=None):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len
