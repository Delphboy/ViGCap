import torch

@torch.jit.export
def create_knn(patch_embeddings: torch.Tensor, k: int) -> torch.Tensor:
    """
    Takes patch embeddings and returns an adjacency matrix based on k-nearest neighbors
        :param patch_embeddings: [B, N, D]
        :param k: number of nearest neighbors

        :return: [B, N, N]
    """
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