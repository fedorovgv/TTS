import logging
import typing as tp

import torch
import torch.nn as nn

from model.encdec import Encoder, Decoder
from model.varianceadaptor import VarianceAdaptor
from configs import FastSpeechConfig, MelSpectrogramConfig

logging.basicConfig(level=logging.DEBUG)


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


class FastSpeech2(nn.Module):
    """ FastSpeech 2 """
    def __init__(self, model_config: FastSpeechConfig, mel_config: MelSpectrogramConfig) -> None:
        super().__init__()

        self.encoder = Encoder(
            model_config,
        )
        self.variance_adaptor = VarianceAdaptor(
            model_config,
        )
        self.decoder = Decoder(
            model_config,
        )
        
        self.mel_linear = nn.Linear(
            model_config.decoder_dim, mel_config.num_mels,
        )

    @staticmethod
    def mask_tensor(
        mel_output: torch.Tensor, 
        position: torch.Tensor, 
        mel_max_length: int,
    ) -> torch.Tensor:
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(
        self, 
        src_seq: torch.Tensor, 
        src_pos: torch.Tensor, 
        mel_pos: tp.Optional[torch.Tensor] = None, 
        mel_max_length: tp.Optional[int] = None, 
        length_target: tp.Optional[torch.Tensor] = None, 
        pitch_target: tp.Optional[torch.Tensor] = None,
        energy_target: tp.Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ) -> tp.Tuple[torch.Tensor, ...]:
        """
        src_seq - decoded text
        src_pos - pos mask
        mel_pos - mel mask
        length_target - duration target
        alpha - speed control
        beta - pitch control
        gamma - energy control
        """
        x, non_pad_mask = self.encoder(src_seq, src_pos)
        if self.training:
            (
                output, 
                log_duration_prediction, 
                pitch_prediction, 
                energy_prediction, 
                mel_len,
            ) = self.variance_adaptor(
                x=x, 
                duration_target=length_target, 
                max_len=mel_max_length, 
                pitch_target=pitch_target, 
                energy_target=energy_target, 
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )
            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(
                output, mel_pos, mel_max_length,
            )
            output = self.mel_linear(output)
            return (
                output, 
                log_duration_prediction,
                pitch_prediction,
                energy_prediction
            )
        else:
            (
                output, 
                log_duration_prediction, 
                pitch_prediction, 
                energy_prediction, 
                mel_len,
            ) = self.variance_adaptor(
                x=x,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )
            mel_pos = get_mask_from_lengths(mel_len)
            output = self.decoder(output, mel_pos)
            output = self.mel_linear(output)
            return (
                output, 
                log_duration_prediction,
                pitch_prediction,
                energy_prediction
            )
