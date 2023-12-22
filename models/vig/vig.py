import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.vig.gnn import GraphAttentionNetwork, GraphConvolutionalNetwork, SagPool
from models.vig.graph_functions import create_knn


class ImageToGridFeatures(nn.Module):
    # create a pretrained ViT model
    def __init__(self, hidden_dim=192) -> None:
        super().__init__()
        self.vit = torchvision.models.vit_b_16(
            weights=torchvision.models.ViT_B_16_Weights.DEFAULT
        )  # .conv_proj

        # Remove the last layers of ViT so that we can extract the features of the tokens
        # Rather than the classification token
        self.vit = nn.Sequential(*list(self.vit.children())[:-2])

        # Freeze all layers of the ViT model
        for param in self.vit.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(768, hidden_dim)

    def forward(self, x):
        B, C, H, W = x.size()
        x = self.vit(x)
        x = x.reshape(B, x.shape[1], -1)
        x = x.transpose(1, 2)
        x = self.linear(x)
        return x


class ImageToGridFeaturesUntrained(nn.Module):
    def patcher(self, x, grid_size=(16, 16)):
        B, C, H, W = x.size()
        x = x.unfold(2, grid_size[0], grid_size[0]).unfold(
            3, grid_size[1], grid_size[1]
        )

        x = x.reshape(B, C, -1, grid_size[0], grid_size[1])  # [B, 3, 14*14, 16, 16]
        x = x.permute(0, 2, 1, 3, 4)  # [B, 196, 3, 16, 16]
        x = x.reshape(-1, C, grid_size[0], grid_size[1])  # [B*196, 3, 16, 16]
        return x

    def __init__(self, img_size=224, in_dim=3, hidden_dim=192, act="relu"):
        super().__init__()

        # We want to patch the image using patcher
        # We then want to flatten the patches and pass them through a linear layer
        # We then want to prepend a learnable embedding to each patch, similar to BERT's [CLS] token
        self.pather = self.patcher
        self.flattener = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(in_dim * 16 * 16, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(act),
        )
        self.embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.positional_embedding = nn.Parameter(
            torch.randn(1, 14 * 14 + 1, hidden_dim)
        )

    def forward(self, x):
        patches = self.patcher(x)
        patches = self.flattener(patches)
        patches = self.linear(patches)
        patches = patches.reshape(x.shape[0], -1, patches.shape[-1])
        # prepend the embedding to each patch
        patches = torch.cat(
            [self.embedding.repeat(patches.shape[0], 1, 1), patches], dim=1
        )
        patches = patches + self.positional_embedding
        return patches


class VigStem(nn.Module):
    """Image to Visual Word Embedding
    Original STEM algorithm from VisionGNN
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, img_size=224, in_dim=3, hidden_dim=192, act="relu"):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim // 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(act),
            nn.Conv2d(hidden_dim // 8, hidden_dim // 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(act),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(act),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(act),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
        )
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)

    @torch.jit.export
    def forward(self, x):
        x = self.convs(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)
        x = x.transpose(1, 2)

        return x


class ImageToPatchFeatures2(nn.Module):
    def __init__(self, img_size=224, in_dim=3, hidden_dim=192, act="relu"):
        super().__init__()
        resnet = torchvision.models.resnet101(
            weights=torchvision.models.ResNet101_Weights.DEFAULT
        )

        # Freeze all layers
        # for param in resnet.parameters():
        # param.requires_grad = False

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, hidden_dim)

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
        self.resnet.eval()
        with torch.no_grad():
            x = self.resnet(x)
        # Now we have a tensor of shape [B*196, 2048, 1, 1]

        x = x.reshape(B, 196, -1)

        # Now we have a tensor of shape [B, 196, 2048]
        x = self.linear(x)
        return x


class ImageToFeatures(nn.Module):
    def __init__(self, img_size=224, num_channels=3, hidden_dim=768, act="relu"):
        super().__init__()
        patch_size = (16, 16)
        patch_height, patch_width = patch_size
        num_heads = 16
        num_layers = 12

        # Calculate the number of patches and patch dimensions
        self.num_patches = (img_size // patch_height) * (img_size // patch_width)
        self.patch_embed = nn.Conv2d(
            num_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        # Positional embedding
        self.positional_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, hidden_dim)
        )

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers,
        )

    def forward(self, x):
        # Extract patches from the input image
        patches = self.patch_embed(x)

        # Rearrange patches for input to the transformer
        patches = patches.view(x.shape[0], self.num_patches, -1)

        # Add positional embeddings
        patches = patches + self.positional_embedding

        # Pass the patches through the transformer
        patches = self.transformer_encoder(patches)

        # Output the patch features
        patch_features = patches

        return patch_features


class Vig(nn.Module):
    def __init__(self, args) -> None:
        super(Vig, self).__init__()
        self.k = args.k
        self.gat_1 = GraphAttentionNetwork(
            args.patch_feature_size, args.gnn_emb_size, dropout=args.dropout
        )
        # self.sag_pool_1 = SagPool(args.gnn_emb_size, args.sag_ratio)
        # self.gat_2 = GraphAttentionNetwork(args.gnn_emb_size, args.meshed_emb_size, 1)
        # self.sag_pool_2 = SagPool(args.meshed_emb_size, args.sag_ratio)

    @torch.jit.export
    def forward(self, superpixel_features: torch.Tensor) -> torch.Tensor:
        # create adjacency matrix
        adj_mat = create_knn(superpixel_features, k=self.k)
        # adj_mat = create_region_adjacency_graph(patch_embeddings)

        # # pass through GAT
        superpixel_features = self.gat_1(superpixel_features, adj_mat)
        # patch_embeddings, adj_mat = self.sag_pool_1(patch_embeddings, adj_mat)

        # patch_embeddings = self.gat_2(patch_embeddings, adj_mat)
        # patch_embeddings, adj_mat = self.sag_pool_2(patch_embeddings, adj_mat)

        return superpixel_features
