import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # self.layer_2 = GraphAttentionLayer(
        #     out_features,
        #     out_features,
        #     n_heads,
        #     is_concat,
        #     dropout,
        #     leaky_relu_negative_slope,
        # )
        # self.activation_2 = nn.LeakyReLU(leaky_relu_negative_slope)

        # self.sag_pool = SagPool(out_features * n_heads, pool_ratio)

    @torch.jit.export
    def forward(
        self,
        x: torch.Tensor,
        adj_mat: torch.Tensor,
    ) -> torch.Tensor:
        x = self.layer_1(x, adj_mat)
        x = self.activation_1(x)
        # x = self.layer_2(x, adj_mat)
        # x = self.activation_2(x)

        # x, adj_mat = self.sag_pool(x, adj_mat)

        return x



class GraphConvolutionalNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        opt=None,
    ) -> None:
        super(GraphConvolutionalNetwork, self).__init__()
        self.layer_1 = GraphConvolutionalLayer(in_features, out_features)
        self.activation_1 = nn.ReLU()

    @torch.jit.export
    def forward(
        self,
        x: torch.Tensor,
        adj_mat: torch.Tensor,
    ) -> torch.Tensor:
        x = self.layer_1(x, adj_mat)
        x = self.activation_1(x)

        return x