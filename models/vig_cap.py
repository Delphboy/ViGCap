from typing import Optional

import torch
from torch import nn

from models.vig.pyramid_vig import (
    pvig_b_224_gelu,
    pvig_m_224_gelu,
    pvig_s_224_gelu,
    pvig_ti_224_gelu,
)
from models.vig.vig import vig_b_224_gelu, vig_s_224_gelu, vig_ti_224_gelu

from .captioning_model import CaptioningModel


class VigCap(CaptioningModel):
    def __init__(
        self,
        bos_idx,
        encoder,
        decoder,
        dropout:Optional[float]=0.5,
        vig_type: Optional[str] = "default",
        vig_size: Optional[str] = "tiny",
        n_blocks: Optional[int] = 16,
    ):
        super(VigCap, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.register_state("enc_output", None)
        self.register_state("mask_enc", None)
        self.init_weights()
        self.dropout = nn.Dropout(dropout)

        if vig_type == "default":
            if vig_size == "base":
                self.vig = vig_b_224_gelu(drop_rate=dropout, n_blocks=n_blocks)
            elif vig_size == "small":
                self.vig = vig_s_224_gelu(drop_rate=dropout, n_blocks=n_blocks)
            else:
                self.vig = vig_ti_224_gelu(drop_rate=dropout, n_blocks=n_blocks)
        elif vig_type == "pyramid":
            if vig_size == "base":
                self.vig = pvig_b_224_gelu(drop_rate=dropout)
            elif vig_size == "small":
                self.vig = pvig_s_224_gelu(drop_rate=dropout)
            elif vig_size == "medium":
                self.vig = pvig_m_224_gelu(drop_rate=dropout)
            else:
                self.vig = pvig_ti_224_gelu(drop_rate=dropout)
        else:
            raise ValueError(f"vig_type {vig_type} not supported")

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @torch.jit.export
    def forward(self, images, seq):
        # Get a set of image features from ViG
        input = self.vig(images)
        # input = self.dropout(input)

        # Encode the image features
        enc_output, mask_enc = self.encoder(input)
        # enc_output = self.dropout(enc_output)

        # Meshed decoder
        dec_output = self.decoder(seq, enc_output, mask_enc)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device), None, None]

    @torch.jit.ignore
    def step(self, t, prev_output, visual, seq, mode="teacher_forcing", **kwargs):
        it = None
        if mode == "teacher_forcing":
            raise NotImplementedError
        elif mode == "feedback":
            if t == 0:
                self.enc_output, self.mask_enc = self.encoder(visual)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = (
                        visual[0]
                        .data.new_full((visual[0].shape[0], 1), self.bos_idx)
                        .long()
                    )
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc)
