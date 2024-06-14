import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool

class Encoder(nn.Module):
    def __init__(self, inp_dim, embed_dim=10, h_nodes=128, dropout=0.2, num_heads=4, num_layers=1):
        super(Encoder, self).__init__()
        self.gat = GATv2Conv(inp_dim, h_nodes, heads=num_heads, concat=True, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(h_nodes * num_heads, h_nodes),
            nn.ReLU(),
            nn.Linear(h_nodes, embed_dim)
        )

    def forward(self, x, edge_index, batch):
        x = self.gat(x, edge_index)
        x = global_mean_pool(x, batch)  # Global pooling
        x = self.mlp(x)
        return x