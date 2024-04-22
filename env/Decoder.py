import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(
        self,
        out_dim,
        embed_dim=10,
        h_nodes=128,
        dropout=0.2,
        scale=2,
        num_layers=1,
        load_path=None,
    ):
        super(Decoder, self).__init__()
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.h_nodes = h_nodes
        self.scale = scale
        self.num_layers = num_layers
        self.fc1 = nn.Linear(self.embed_dim, h_nodes)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(h_nodes // (scale**i), h_nodes // (scale ** (i + 1))),
                    nn.ReLU(inplace=True),
                )
                for i in range(num_layers)
            ]
        )
        self.fc2 = nn.Linear(h_nodes // scale ** (num_layers), self.out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        if load_path:
            self.load_state_dict(torch.load(load_path, map_location="cpu"))
            self.eval()
            print("Decoder model loaded from: ", load_path)

    def forward(self, x):

        with torch.no_grad():
            x = self.relu(self.fc1(x))
            for layer in self.layers:
                x = layer(x)
            x = self.fc2(x)
            x = self.relu(x)
            return x
