import torch                                            #TODO: Change this code to works with vector representation instead of graph
import torch.nn as nn                                   #TODO: Have the decoder_model be saved at env/models when training is done
import torch.nn.functional as F                         #TODO: Have the codebook be saved at env/models when training is done
import numpy as np       
from model import SVQVAE
from dataset import VQVAE_Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datetime import datetime

def encode_points(model, data, labels, device):
    model.eval()
    with torch.no_grad():
        z = model.encoder(data.to(device))  # Encode data points
        _, quantized_z, _, _ = model.quantizer(z, labels.to(device))  # Quantize the encoded vectors
    return quantized_z

# Function to perform t-SNE and plot the results
def plot_with_tsne(latent_vectors, labels, codebook_vectors):
    tsne = TSNE(n_components=2, random_state=42)
    all_vectors = np.vstack([latent_vectors, codebook_vectors])  # Combine latent and codebook vectors
    all_vectors_2d = tsne.fit_transform(all_vectors)

    latent_vectors_2d = all_vectors_2d[:-len(codebook_vectors)]
    codebook_vectors_2d = all_vectors_2d[-len(codebook_vectors):]

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(latent_vectors_2d[:, 0], latent_vectors_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.scatter(codebook_vectors_2d[:, 0], codebook_vectors_2d[:, 1], color='black', marker='x')  # Codebook vectors in black

    # Create a legend for the labels
    unique_labels = np.unique(labels)
    colors = scatter.cmap(np.linspace(0, 1, len(unique_labels)))
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(int(label)), markersize=10, markerfacecolor=color) for label, color in zip(unique_labels, colors)]
    plt.legend(handles=legend_handles, title='Labels')

    plt.title('t-SNE Visualization of Latent Vectors and Codebook Vectors')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.savefig('latent_space.png')

def train(args):
    dataset = VQVAE_Dataset(dataFilePath=args.filePath)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device  = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    x_dim = dataset.data.shape[1]
    model = SVQVAE(x_dim,args.embed_dim,dropout=args.dropout,num_embeddings=args.num_embeddings,
                commitment_cost=args.commitment_cost,divergence_cost=args.divergence_cost,
                h_nodes=args.h_nodes,scale=args.scale,num_layers=args.num_layers).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.wd, amsgrad=False)

    model.train()
    train_res_recon_error = []
    train_res_perplexity = []
    close_indices = None

    for epoch in tqdm(range(args.epochs)):

        for i, batch in enumerate(train_loader):  # Iterate over all batches in the dataset
            batch_x,batch_y = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()

            # noise = torch.rand(batch.size()).to(device)
            vq_loss, data_recon, perplexity, close_indices = model(batch_x,batch_y)
            recon_error = F.mse_loss(data_recon, batch_x)
            loss = recon_error + vq_loss
            loss.backward()

            optimizer.step()

            train_res_recon_error.append(recon_error.item())
            train_res_perplexity.append(perplexity.item())

            if (i+1) % 2 == 0:
                print('%d iterations of epochs %d/%d' % (i+1,epoch + 1, args.epochs))
                print('Average recon_error: %.3f' % np.mean(train_res_recon_error[-10:]))
                print('Average perplexity: %.3f' % np.mean(train_res_perplexity[-10:]))
                print()

    '''Visualizng the latent space'''
    # Encode the data points
    quantized_z = encode_points(model, dataset.data, dataset.labels,device)
    # Get codebook vectors from the quantizer
    codebook_vectors = model.quantizer.embeddings.weight.data.cpu().numpy()
    # Plot using t-SNE
    plot_with_tsne(quantized_z.cpu().numpy(), dataset.labels.cpu().numpy(), codebook_vectors)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    '''storing the latent representations of the architectures for rl render function'''
    quantized_z_np = quantized_z.cpu().numpy()
    labels_np = dataset.labels.cpu().numpy()
    np.save(f'../render/architectures_trained_on_{timestamp}.npy', quantized_z_np)
    np.save(f'../render/labels_{timestamp}.npy', labels_np)

    '''storing the trained decoder and codebook weights for the rl agent training'''
    model.eval()
    decoder = model.decoder
    torch.save(decoder.state_dict(), f'../models/decoder_model_{timestamp}.pth')
    quantizer = model.quantizer
    torch.save(quantizer.embeddings.weight.data, f'../models/codebook_{timestamp}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="vqvae train")
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--filePath',default='../../data/dataset_cifar10_v1.csv',type=str, help='path to the nasbench data')
    parser.add_argument('--lr',default=0.0006938,type=float)
    parser.add_argument('--wd',default=0.0002155,type=float)
    parser.add_argument('--embed_dim',default=8,type=int)
    parser.add_argument('--num_embeddings',default=14,type=int)
    parser.add_argument('--commitment_cost',default=0.25,type=float)
    parser.add_argument('--divergence_cost',default=0.25,type=float)
    parser.add_argument('--h_nodes',default=512,type=float)
    parser.add_argument('--scale',default=2,type=int)
    parser.add_argument('--num_layers',default=5,type=int)
    parser.add_argument('--dropout',default=0.2,type=float)

    args = parser.parse_args()

    print("Start the training")

    train(args=args)