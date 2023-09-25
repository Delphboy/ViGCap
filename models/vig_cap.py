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
        vig_type: Optional[str] = "default",
        vig_size: Optional[str] = "tiny",
    ):
        super(VigCap, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.register_state("enc_output", None)
        self.register_state("mask_enc", None)
        self.init_weights()

        if vig_type == "default":
            if vig_size == "base":
                self.vig = vig_b_224_gelu()
            elif vig_size == "small":
                self.vig = vig_s_224_gelu()
            else:
                self.vig = vig_ti_224_gelu()
        elif vig_type == "pyramid":
            if vig_size == "base":
                self.vig = pvig_b_224_gelu()
            elif vig_size == "small":
                self.vig = pvig_s_224_gelu()
            elif vig_size == "medium":
                self.vig = pvig_m_224_gelu()
            else:
                self.vig = pvig_ti_224_gelu()
        else:
            raise ValueError(f"vig_type {vig_type} not supported")

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, seq, *args):
        # Get a set of image features from ViG
        input = self.vig(images)

        # Encode the image features
        enc_output, mask_enc = self.encoder(input)

        # Meshed decoder
        dec_output = self.decoder(seq, enc_output, mask_enc)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device), None, None]

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


# class TransformerEnsemble(CaptioningModel):
#     def __init__(self, model: Transformer, weight_files):
#         super(TransformerEnsemble, self).__init__()
#         self.n = len(weight_files)
#         self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
#         for i in range(self.n):
#             state_dict_i = torch.load(weight_files[i])["state_dict"]
#             self.models[i].load_state_dict(state_dict_i)

#     def step(self, t, prev_output, visual, seq, mode="teacher_forcing", **kwargs):
#         out_ensemble = []
#         for i in range(self.n):
#             out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
#             out_ensemble.append(out_i.unsqueeze(0))

#         return torch.mean(torch.cat(out_ensemble, 0), dim=0)
