import copy

import torch
import torch.nn.functional as F
from torch import nn

from models.meshed.captioning_model import CaptioningModel
from models.meshed.containers import ModuleList
from models.vig.vig import Vig


class VigCap(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder, args):
        super(VigCap, self).__init__()
        self.bos_idx = bos_idx
        self.vig = Vig(args)
        # This linear can be replaced if the last GAT layer output is same dim as M2
        self.linear = nn.Linear(args.gnn_emb_size, args.meshed_emb_size, bias=False)
        self.encoder = encoder
        self.decoder = decoder
        self.dropout = nn.Dropout(args.dropout)
        self.register_state("enc_output", None)
        self.register_state("mask_enc", None)
        # self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, seq, *args):
        images = self.vig(images)
        images = self.dropout(images)
        # images = self.linear(images)

        enc_output, mask_enc = self.encoder(images)
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
                visual = self.vig(visual)
                # visual = self.linear(visual)
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


class VigCapEnsemble(CaptioningModel):
    def __init__(self, model: VigCap, weight_files):
        super(VigCapEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])["state_dict"]
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode="teacher_forcing", **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)
