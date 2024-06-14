import torch
from torch_geometric.data import DataLoader
from model import SVQVAE
from dataset import NASBenchDataset
import argparse

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

OPERATIONS = [INPUT, OUTPUT, CONV1X1, CONV3X3, MAXPOOL3X3]

def train(args):
    data = torch.load(args.data_path)
    dataset = NASBenchDataset(nasbench_data=data)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    x_dim = len(OPERATIONS)
    model = SVQVAE(x_dim,args.embed_dim,num_embeddings=args.num_embeddings,
                   commitment_cost=args.commitment_cost,divergence_cost=args.divergence_cost,
                   h_nodes=args.h_nodes,num_heads=args.num_heads)

if __name__ == 'main':
    parser = argparse.ArgumentParser(description="vqvae train")
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--data_path',default='./nasbench_dataset.pt',type=str, help='path to the nasbench data')
    parser.add_argument('--lr',default=1e-4,type=float)
    parser.add_argument('--wd',default=1e-3,type=float)
    parser.add_argument('--embed_dim',default=10,type=int)
    parser.add_argument('--num_embeddings',default=19,type=int)
    parser.add_argument('--commitment_cost',default=0.25,type=float)
    parser.add_argument('--divergence_cost',default=0.1,type=float)
    parser.add_argument('--h_nodes',default=32,type=float)
    parser.add_argument('--num_heads',default=4,type=int)

    args = parser.parse_args()

    train(args=args)