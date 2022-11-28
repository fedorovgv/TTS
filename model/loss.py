import typing as tp

import torch
import torch.nn as nn


class FastSpeechLoss2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def forward(
        self, 
        mel_target: torch.Tensor,
        mel_predicted: torch.Tensor,
        mel_pos: torch.Tensor,
        duration_target: torch.Tensor,
        log_duration_predicted: torch.Tensor,
        pitch_target: torch.Tensor,
        pitch_predicted: torch.Tensor,
        energy_target: torch.Tensor,
        energy_predicted: torch.Tensor,
        src_pos: torch.Tensor,
    ) -> tp.Tuple[int, ...]:

        # for flash attention
        mel_target.requires_grad = False
        duration_target.requires_grad = False
        pitch_target.requires_grad = False
        energy_target.requires_grad = False
        
        mask = mel_pos.bool().unsqueeze(-1).expand(mel_predicted.size())
        mel_loss = self.mae(
            mel_target.masked_select(mask.bool()),
            mel_predicted.masked_select(mask.bool()), 
        )
        
        log_duration_target = torch.log(duration_target.float() + 1.0)
        duration_loss = self.mse(
            log_duration_target.masked_select(src_pos.bool()),
            log_duration_predicted.masked_select(src_pos.bool())
        )

        pitch_loss = self.mse(
            pitch_predicted.masked_select(mel_pos.bool()), 
            pitch_target.masked_select(mel_pos.bool()),
        )
        
        energy_loss = self.mse(
            energy_predicted.masked_select(mel_pos.bool()), 
            energy_target.masked_select(mel_pos.bool()),
        )
        
        total_loss = (mel_loss + duration_loss + pitch_loss + energy_loss)

        return total_loss, mel_loss, duration_loss, pitch_loss, energy_loss
