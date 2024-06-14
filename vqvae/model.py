import torch.nn as nn
from decoder import Decoder
from encoder import Encoder
from quantizer import VectorQuantizerClass

class SVQVAE(nn.Module):
  def __init__(self,x_dim,embed_dim=10,dropout=0.2,num_embeddings=50,
               commitment_cost=0.25,divergence_cost=0.1,h_nodes=128,scale=2,num_heads=4,num_layers=1):
    super(SVQVAE,self).__init__()
    self.encoder = Encoder(x_dim,embed_dim,dropout=dropout,h_nodes=h_nodes,num_heads=num_heads,
                           num_layers=num_layers)
    self.quantizer = VectorQuantizerClass(num_embeddings,embed_dim,commitment_cost,divergence_cost)
    self.decoder = Decoder(x_dim,embed_dim,h_nodes=h_nodes,dropout=dropout,
                           num_layers=num_layers)

  def forward(self,x,y):
    node = x.x
    edges = x.edge_index
    z = self.encoder(node,edges)
    loss, quantized_z, perplexity, close_indices = self.quantizer(z,y)
    node_recon, adj_recon = self.decoder(quantized_z)
    return loss, node_recon, adj_recon, perplexity, close_indices