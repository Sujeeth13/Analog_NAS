import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.utils import unbatch_edge_index
from model import SVQVAE
from dataset import NASBenchDataset
from tqdm import tqdm
import argparse

VERTICES = 7
INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

OPERATIONS = [INPUT, OUTPUT, CONV1X1, CONV3X3, MAXPOOL3X3]

def train(args):
    data = torch.load(args.data_path)
    dataset = NASBenchDataset(nasbench_data=data)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    device  = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    x_dim = len(OPERATIONS)
    model = SVQVAE(x_dim,args.embed_dim,num_embeddings=args.num_embeddings,
                   commitment_cost=args.commitment_cost,divergence_cost=args.divergence_cost,
                   h_nodes=args.h_nodes,num_heads=args.num_heads).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    criterion_reconstruction = nn.CrossEntropyLoss()
    pos_weight = torch.tensor([0.8], dtype=torch.float32).to(device)
    criterion_edge_reconstruction = nn.BCEWithLogitsLoss(pos_weight = pos_weight)

    model.train()
    train_adj_recon_error = []
    train_node_recon_error = []
    train_res_perplexity = []
    close_indices = None

    for epoch in tqdm(range(args.epochs)):
        for i, batch in enumerate(train_loader):  # Iterate over all batches in the dataset
            batch = batch.to(device)
            optimizer.zero_grad()

            # noise = torch.rand(batch.size()).to(device)
            vq_loss, node_recon, adj_recon, perplexity, close_indices = model(batch)

            # Reconstruction loss for node features
            loss_reconstruction = criterion_reconstruction(node_recon.view(-1, 5), batch.x.argmax(dim=1))

            # Edge reconstruction loss
            original_adjacency = []
            for j in range(batch.batch.max().item() + 1):
                adj = torch.zeros((VERTICES,VERTICES),device = device)
                edges = unbatch_edge_index(batch.edge_index,batch.batch)
                edges = edges[j]
                adj[edges[0],edges[1]] = 1
                original_adjacency.append(adj)

            original_adjacency = torch.stack(original_adjacency).to(device)
            loss_edge_reconstruction = criterion_edge_reconstruction(adj_recon, original_adjacency)

            # Total loss
            loss = loss_reconstruction + loss_edge_reconstruction + vq_loss

            loss.backward()

            optimizer.step()

            train_node_recon_error.append(loss_reconstruction.item())
            train_adj_recon_error.append(loss_edge_reconstruction.item())
            train_res_perplexity.append(perplexity.item())

            if (i+1) % 2 == 0:
                print('%d iterations of epochs %d/%d' % (i+1,epoch + 1, args.epochs))
                print('Average node_recon_error: %.3f' % np.mean(train_node_recon_error[-10:]))
                print('Average adj_recon_error: %.3f' % np.mean(train_adj_recon_error[-10:]))
                print('Average perplexity: %.3f' % np.mean(train_res_perplexity[-10:]))
                print()
    
    model.eval()
    for i, batch in enumerate(val_loader):  # Iterate over all batches in the dataset
        batch = batch.to(device)

        # noise = torch.rand(batch.size()).to(device)
        vq_loss, node_recon, adj_recon, perplexity, close_indices = model(batch)

        # Reconstruction loss for node features
        loss_reconstruction = criterion_reconstruction(node_recon.view(-1, 5), batch.x.argmax(dim=1))

        # Edge reconstruction loss
        original_adjacency = []
        for j in range(batch.batch.max().item() + 1):
            adj = torch.zeros((VERTICES,VERTICES),device = device)
            edges = unbatch_edge_index(batch.edge_index,batch.batch)
            edges = edges[j]
            adj[edges[0],edges[1]] = 1
            original_adjacency.append(adj)

        original_adjacency = torch.stack(original_adjacency).to(device)
        loss_edge_reconstruction = criterion_edge_reconstruction(adj_recon, original_adjacency)

        # Total loss
        loss = loss_reconstruction + loss_edge_reconstruction + vq_loss

        print(adj_recon)
        print(original_adjacency)
        print("--------------")
        print(node_recon)
        print(batch.x)
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="vqvae train")
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--data_path',default='nasbench_dataset.pt',type=str, help='path to the nasbench data')
    parser.add_argument('--lr',default=1e-4,type=float)
    parser.add_argument('--wd',default=1e-3,type=float)
    parser.add_argument('--embed_dim',default=10,type=int)
    parser.add_argument('--num_embeddings',default=20,type=int)
    parser.add_argument('--commitment_cost',default=0.25,type=float)
    parser.add_argument('--divergence_cost',default=0.1,type=float)
    parser.add_argument('--h_nodes',default=32,type=float)
    parser.add_argument('--num_heads',default=4,type=int)

    args = parser.parse_args()

    print("Start the training")

    train(args=args)