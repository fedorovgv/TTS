import torch
import typing as tp

from model.transformer import (
    MultiHeadAttention,
    PositionWiseFeedForward,
)

T = tp.TypeVar("T", int, tuple)


class FFTBlock(torch.nn.Module):
    """ FFT Block """
    def __init__(
        self, 
        d_model: int, 
        d_inner: int, 
        n_head: int, 
        d_k: int, 
        d_v: int, 
        kernel: T, 
        padding: T, 
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.slf_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout,
        )
        self.pos_ffn = PositionWiseFeedForward(
            d_model,
            d_inner,
            kernel=kernel,
            padding=padding,
            dropout=dropout,
        )

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask,
        )
        if non_pad_mask is not None:
            enc_output *= non_pad_mask
        enc_output = self.pos_ffn(enc_output)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask
        return enc_output, enc_slf_attn
