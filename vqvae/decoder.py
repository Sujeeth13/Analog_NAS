import torch
import torch.nn as nn

VERTICES = 7 # max number of nodes in the computation graph of the model generated

class Decoder(nn.Module):
    def __init__(self, out_dim,num_nodes, embed_dim=10, h_nodes=128, dropout=0.2, num_layers=1):
        super(Decoder, self).__init__()
        self.num_nodes = VERTICES
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4* h_nodes),
            nn.ReLU(),
            nn.Linear(4* h_nodes, 2* h_nodes),
            nn.ReLU(),
            nn.Linear(2* h_nodes, h_nodes),
            nn.ReLU(),
            nn.Linear(h_nodes, out_dim * VERTICES),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.mlp(x)
        node_features = x.view(batch_size, self.num_nodes, -1)  # Reshape to (B, num_nodes, out_dim)
        adjacency_matrix = torch.matmul(node_features, node_features.transpose(1, 2))
        return node_features, adjacency_matrix