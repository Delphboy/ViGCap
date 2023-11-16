import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.vig.gnn import GraphAttentionNetwork, GraphConvolutionalNetwork, SagPool
from models.vig.graph_functions import create_knn


class ImageToPatchFeatures(nn.Module):
    """Image to Visual Word Embedding
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
        self.image_to_patch_features = ImageToPatchFeatures(
            hidden_dim=args.patch_feature_size
        )
        self.gat_1 = GraphAttentionNetwork(
            args.patch_feature_size, args.gnn_emb_size, 1
        )
        self.sag_pool_1 = SagPool(args.gnn_emb_size, args.sag_ratio)
        self.gat_2 = GraphAttentionNetwork(args.gnn_emb_size, args.meshed_emb_size, 1)
        self.sag_pool_2 = SagPool(args.meshed_emb_size, args.sag_ratio)

    @torch.jit.export
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # create patch embeddings
        patch_embeddings = self.image_to_patch_features(image)
        # create adjacency matrix
        adj_mat = create_knn(patch_embeddings, k=self.k)

        # pass through GAT
        patch_embeddings = self.gat_1(patch_embeddings, adj_mat)
        patch_embeddings, adj_mat = self.sag_pool_1(patch_embeddings, adj_mat)

        patch_embeddings = self.gat_2(patch_embeddings, adj_mat)
        patch_embeddings, adj_mat = self.sag_pool_2(patch_embeddings, adj_mat)

        return patch_embeddings


class VigResnet(nn.Module):
    def __init__(self, emb_size: int) -> None:
        super(VigResnet, self).__init__()
        self.gnn = GraphAttentionNetwork(emb_size, emb_size // 4, 1)

    @torch.jit.export
    def forward(self, resnet_features: torch.Tensor) -> torch.Tensor:
        # create adjacency matrix
        adj_mat = create_knn(resnet_features, k=5)

        # pass through GAT
        resnet_features = self.gnn(resnet_features, adj_mat)

        return resnet_features
