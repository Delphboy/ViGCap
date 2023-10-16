from typing import List, Optional

import torch
import torch.nn as nn
import torchvision

import utils
from models.captioning_model import CaptioningModel


class GraphAttentionLayer(nn.Module):
    """
    ## Graph attention layer based on: https://nn.labml.ai/graphs/gat/index.html
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,  # TODO: Take in the --n_heads argument
        is_concat: bool = True,
        dropout: float = 0.6,
        leaky_relu_negative_slope: float = 0.2,
    ):
        """
        * in_features: number of input features per node
        * out_features: number of output features per node
        * n_heads: number of attention heads
        * is_concat: whether the multi-head results should be concatenated or averaged
        * dropout: dropout probability
        * leaky_relu_negative_slope: negative slope for leaky relu activation
        """
        super(GraphAttentionLayer, self).__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)

        # Softmax to compute attention weight
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    @torch.jit.export
    def forward(
        self,
        h: torch.Tensor,
        adj_mat: torch.Tensor,
    ):
        """
        * h, is the input node embeddings of shape [n_nodes, in_features].
        * adj_mat is the adjacency matrix of shape [n_nodes, n_nodes, n_heads].
        We use shape [n_nodes, n_nodes, 1] since the adj is the same for each head
        * edge_attr is the edge attributes of shape [n_edges, edge_attr_dim].

        Adjacency matrix represent the edges (or connections) among nodes.
        adj_mat[i][j] is 1 if there is an edge from node i to node j.
        """
        batch_size = h.shape[0]
        num_nodes = h.shape[1]

        # Add a dimension for the number of heads
        adj_mat = adj_mat.unsqueeze(-1)

        # Add self-connections
        adj_mat = adj_mat + torch.eye(num_nodes).to(adj_mat.device).unsqueeze(-1)

        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == num_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == num_nodes, print(
            adj_mat.shape, num_nodes
        )
        assert adj_mat.shape[3] == 1 or adj_mat.shape[3] == self.n_heads

        g = self.linear(h).view(batch_size, num_nodes, self.n_heads, self.n_hidden)

        g_repeat = g.repeat(1, num_nodes, 1, 1)
        g_repeat_interleave = g.repeat_interleave(num_nodes, dim=1)

        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)

        g_concat = g_concat.view(
            batch_size, num_nodes, num_nodes, self.n_heads, self.n_hidden * 2
        )

        e = self.activation(self.attn(g_concat))
        e = e.squeeze(-1)

        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == num_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == num_nodes, print(
            adj_mat.shape, num_nodes
        )
        assert adj_mat.shape[3] == 1 or adj_mat.shape[3] == self.n_heads

        e = e.masked_fill(adj_mat == 0, -1000)
        a = self.softmax(e)
        a = self.dropout(a)

        attn_res = torch.einsum("bijh,bjhf->bihf", a, g)

        if self.is_concat:
            return attn_res.reshape(batch_size, num_nodes, self.n_hidden * self.n_heads)
        else:
            return attn_res.mean(dim=2)


class GraphConvolutionalLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(GraphConvolutionalLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    @torch.jit.export
    def forward(
        self,
        h: torch.Tensor,
        adj_mat: torch.Tensor,
    ) -> torch.Tensor:
        h = self.linear(h)
        h = torch.matmul(adj_mat, h)

        return h


class SagPool(nn.Module):
    # An implementation of the Self-Attention Graph Pooling layer
    # based on https://arxiv.org/pdf/1904.08082.pdf
    # Reduce the number of nodes in the graph by pooling by a ratio
    def __init__(self, in_channels: int, ratio: int) -> None:
        super(SagPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.linear = nn.Linear(in_channels, 1)
        self.activation = nn.Tanh()

    @torch.jit.export
    def forward(self, nodes: torch.Tensor, adj_mat: torch.Tensor):
        # nodes: [batch_size, num_nodes, in_channels]
        # adj_mat: [batch_size, num_nodes, num_nodes]
        nodes.shape[0]
        num_nodes = nodes.shape[1]

        # Compute the attention weights
        attn_weights = self.linear(nodes).squeeze(-1)  # [batch_size, num_nodes]
        idx = torch.topk(
            attn_weights,
            k=int(num_nodes * self.ratio),
            dim=1,
            largest=True,
            sorted=False,
        )[1]

        # use idx to select the nodes
        nodes = nodes.gather(
            dim=1, index=idx.unsqueeze(-1).expand(-1, -1, nodes.shape[-1])
        )
        # print(f"Reduced from {num_nodes} to {nodes.shape[1]} nodes")
        num_nodes = nodes.shape[1]

        # Reduce the adjacency matrix
        # without using gather
        adj_mat = adj_mat.gather(
            dim=1, index=idx.unsqueeze(-1).expand(-1, -1, num_nodes)
        )  # [batch_size, num_nodes, num_nodes]

        return nodes, adj_mat


class GraphAttentionNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,
        is_concat: bool = True,
        dropout: float = 0.6,
        leaky_relu_negative_slope: float = 0.1,
        opt=None,
    ) -> None:
        super(GraphAttentionNetwork, self).__init__()
        pool_ratio = 0.5
        self.layer_1 = GraphAttentionLayer(
            in_features,
            out_features,
            n_heads,
            is_concat,
            dropout,
            leaky_relu_negative_slope,
        )
        self.activation_1 = nn.LeakyReLU(leaky_relu_negative_slope)

        self.layer_2 = GraphAttentionLayer(
            out_features,
            out_features,
            n_heads,
            is_concat,
            dropout,
            leaky_relu_negative_slope,
        )
        self.activation_2 = nn.LeakyReLU(leaky_relu_negative_slope)

        self.sag_pool = SagPool(out_features * n_heads, pool_ratio)

    @torch.jit.export
    def forward(
        self,
        x: torch.Tensor,
        adj_mat: torch.Tensor,
    ) -> torch.Tensor:
        x = self.layer_1(x, adj_mat)
        x = self.activation_1(x)
        x = self.layer_2(x, adj_mat)
        x = self.activation_2(x)

        x, adj_mat = self.sag_pool(x, adj_mat)

        return x


class ImageToPatchFeatures(nn.Module):
    """Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, img_size=224, in_dim=3, out_dim=192, act="relu"):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 8),
            nn.ReLU(act),
            nn.Conv2d(out_dim // 8, out_dim // 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 4),
            nn.ReLU(act),
            nn.Conv2d(out_dim // 4, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            nn.ReLU(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    @torch.jit.export
    def forward(self, x):
        x = self.convs(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)
        x = x.transpose(1, 2)

        return x


class ImageToPatchFeatures2(nn.Module):
    def __init__(self, img_size=224, in_dim=3, out_dim=192, act="relu"):
        super().__init__()
        resnet = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )

        # Freeze all layers
        # for param in resnet.parameters():
        # param.requires_grad = False

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, out_dim)

    def forward(self, x):
        # TODO: Finish ResNet implementation
        B, C, H, W = x.size()
        # We want to patch the image into 16x16 patches and then run the resnet on each
        # patch so the final result will be of the shape [B, 2048, 14, 14]
        x = x.unfold(2, 16, 16).unfold(3, 16, 16)
        # Now we have a tensor of shape [B, 3, 14, 14, 16, 16]

        # We want to reshape this into [B, 3, 14*14, 16, 16]
        x = x.reshape(B, C, -1, 16, 16)  # [B, 3, 196, 16, 16]
        x = x.permute(0, 2, 1, 3, 4)
        # Now we have a tensor of shape [B, 196, 3, 16, 16]
        # We want to reshape this into [B*196, 3, 16, 16]
        x = x.reshape(-1, C, 16, 16)
        # Now we have a tensor of shape [B*196, 3, 16, 16]
        # self.resnet.eval()
        # with torch.no_grad():
        x = self.resnet(x)
        # Now we have a tensor of shape [B*196, 2048, 1, 1]

        x = x.reshape(B, 196, -1)

        # Now we have a tensor of shape [B, 196, 2048]
        x = self.linear(x)
        return x


# Takes patch embeddings and returns an adjacency matrix
@torch.jit.export
def create_knn(patch_embeddings: torch.Tensor, k: int) -> torch.Tensor:
    # X is of the shape [B, N, D]
    patch_embeddings = patch_embeddings.unsqueeze(2)
    x_t = patch_embeddings.transpose(1, 2)
    dist = torch.norm(patch_embeddings - x_t, dim=-1)  # [B, N, N]

    # Get the k-nearest neighbors
    _, idx = torch.topk(dist, k=k, dim=-1, largest=False)  # [B, N, k]

    # Create the adjacency matrix
    adj_mat = torch.zeros(
        patch_embeddings.shape[0], patch_embeddings.shape[1], patch_embeddings.shape[1]
    ).to(patch_embeddings.device)
    adj_mat = adj_mat.scatter(2, idx, 1)
    return adj_mat


class Vig(nn.Module):
    def __init__(self, emb_size: int) -> None:
        super(Vig, self).__init__()
        self.image_to_patch_features = ImageToPatchFeatures(out_dim=emb_size)
        self.gat = GraphAttentionNetwork(emb_size, emb_size, 1)

    @torch.jit.export
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # create patch embeddings
        patch_embeddings = self.image_to_patch_features(image)

        # create adjacency matrix
        adj_mat = create_knn(patch_embeddings, k=5)

        # pass through GAT
        patch_embeddings = self.gat(patch_embeddings, adj_mat)

        return patch_embeddings


class VigCap(CaptioningModel):
    def __init__(
        self,
        bos_idx,
        encoder,
        decoder,
        emb_size: int,
        dropout: Optional[float] = 0.5,
    ):
        super(VigCap, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.vig = Vig(emb_size)
        self.register_state("enc_output", None)
        self.register_state("mask_enc", None)
        self.init_weights()
        self.dropout = nn.Dropout(dropout)

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @torch.jit.export
    def forward(self, images, captions):
        input = self.vig(images)
        enc_output, mask_enc = self.encoder(input)
        dec_output = self.decoder(captions, enc_output, mask_enc)
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
                visual = self.vig(visual)
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

    def caption(
        self,
        visual,
        max_len: int,
        vocab,
    ):
        # Create a sequence of unk tokens of size b_s, max_len
        b_s = visual.shape[0]
        device = visual.device
        seq = torch.full((b_s, 1), self.bos_idx, dtype=torch.long, device=device)

        for t in range(max_len):
            input = self.vig(visual)
            enc_output, mask_enc = self.encoder(input)
            # captions should be a tensor of unk tokens of size b_s, max_len
            dec_output = self.decoder(seq, enc_output, mask_enc, _is_stateful=False)
            seq = torch.cat([seq, dec_output.argmax(-1)], -1)

            # if t % 2 == 0:
            #     debug(dec_output, vocab)

        return dec_output.argmax(-1), None


def debug(out, vocab):
    # get the top 3 most likely words
    top = out.topk(3, dim=-1).indices
    # convert to words
    top_words = []
    for i in range(top.shape[0]):
        top_words.append([vocab.itos[str(int(i))] for i in top[i]])

    print(top_words)


def decode(out, vocab) -> str:
    return " ".join(
        [
            vocab.itos[str(int(i))]
            for i in out
            # if i != text_field.vocab.stoi["<PAD>"]
            # and i != text_field.vocab.stoi["<EOS>"]
            # and i != text_field.vocab.stoi["<SOS>"]
        ]
    )
