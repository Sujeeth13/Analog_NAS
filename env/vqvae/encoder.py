import torch
import torch.nn as nn

class Encoder(nn.Module):
  def __init__(self,inp_dim,embed_dim=10,h_nodes=128,dropout=0.2,scale=2,num_layers=1):
    super(Encoder,self).__init__()
    self.inp_dim = inp_dim
    self.embed_dim = embed_dim
    self.tanh = nn.Tanh()
    self.dropout = nn.Dropout(dropout)
    self.scale = scale
    self.num_layers = num_layers
    self.h_nodes = h_nodes
    self.relu = nn.ReLU()
    self.layers = nn.ModuleList([nn.Sequential(
                                      nn.Linear(h_nodes//(scale**i),h_nodes//(scale**(i+1))),
                                      nn.ReLU(inplace=True),
                                      # nn.Dropout(dropout)
                                  )
                                  for i in range(num_layers)])
    self.fc1 = nn.Linear(self.inp_dim,h_nodes)
    self.fc2 = nn.Linear(h_nodes//(scale**(num_layers)),self.embed_dim)

  def forward(self,x):
    x = self.relu(self.fc1(x))
    for layer in self.layers:
      x = layer(x)
    x = self.fc2(x)
    return x