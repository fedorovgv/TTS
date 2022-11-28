from dataclasses import dataclass


@dataclass
class FastSpeechConfig:
    vocab_size: int = 300
    max_seq_len: int = 3000

    encoder_dim: int = 256
    encoder_n_layer: int = 4
    encoder_head: int = 2
    encoder_conv1d_filter_size: int = 1024 # PositionWiseFeedForward

    decoder_dim = 256
    decoder_n_layer = 4
    decoder_head = 2
    decoder_conv1d_filter_size = 1024

    fft_conv1d_kernel = (9, 1)
    fft_conv1d_padding = (4, 0)

    variance_predictor_filter_size = 256
    variance_predictor_kernel_size = 3
    dropout = 0.1

    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'

    n_bins: int = 256
    
    pitch_min: float = -0.010687095733886202
    pitch_max: float = 0.06157758246120526
    pitch_mean: float = 127.34142298
    pitch_var: float = 11915.43765944
    
    energy_min: float = 0.08180079609155655
    energy_max: float = 1.09148371219635
    energy_mean: float = -1.2334156013502238
    energy_var: float = 15.09041708561776
