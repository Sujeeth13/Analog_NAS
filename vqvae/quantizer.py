import torch
import torch.nn as nn

class VectorQuantizerClass(nn.Module):
  def __init__(self,num_embeddings=50,embedding_dim=10,commitment_cost=0.25,divergence_cost=0.1):
    super(VectorQuantizerClass,self).__init__()
    self.num_embeddings = num_embeddings
    self.embed_dim = embedding_dim
    self.embeddings = nn.Embedding(self.num_embeddings,self.embed_dim)
    self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
    self.commitment_cost = commitment_cost
    self.divergence_cost = divergence_cost

  def forward(self,x,y):
    # x: [B,D]
    # embeddings: [num_embeddings, embed_dim]

    latent_vectors = x.unsqueeze(1) # [B,D] -> [B,1,D]
    codebook_vectors = self.embeddings.weight.unsqueeze(0) # [num_embeddings, embed_dim] -> [1,num_embeddings, embed_dim]
    distances = torch.norm(latent_vectors - codebook_vectors, dim=2) # [B,num_embeddings]

    encoding_indices = torch.reshape(y,(y.shape[0], 1))
    encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings).to(device)
    encodings.scatter_(1, encoding_indices, 1)

    close_indices = torch.argmin(distances, dim=1).unsqueeze(1)
    close_encodings = torch.zeros(close_indices.shape[0], self.num_embeddings).to(device)
    close_encodings.scatter_(1, close_indices, 1)

    indicator = 1 - (encoding_indices == close_indices).int()
    indicator = indicator.float()

    # Quantize
    quantized = torch.matmul(encodings, self.embeddings.weight)
    close_quantized = torch.matmul(close_encodings, self.embeddings.weight)

    # Loss
    q_latent_loss = torch.mean((quantized - x.detach())**2)
    e_latent_loss = torch.mean((quantized.detach() - x)**2)
    x_latent_loss = torch.mean(indicator * ((close_quantized - x.detach())**2))
    d_latent_loss = torch.mean(indicator * ((close_quantized.detach() - x)**2))
    loss = q_latent_loss + self.commitment_cost * e_latent_loss - x_latent_loss - self.divergence_cost * d_latent_loss

    # to copy gradients from decoder to encoder
    quantized = x + (quantized - x).detach()

    avg_probs = torch.mean(encodings, dim=0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

    return loss, quantized, perplexity, close_indices