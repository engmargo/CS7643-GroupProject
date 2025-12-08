import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero

class GNNEncoder(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # SAGEConv with (-1, -1) allows lazy initialization for bipartite graphs
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class DotProductDecoder(nn.Module):
    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        # Get embeddings
        user_emb = z_dict['User'][row]
        prod_emb = z_dict['Product'][col]
        
        # Normalize embeddings to ensure we calculate Cosine Similarity.
        user_emb = F.normalize(user_emb, p=2, dim=-1)
        prod_emb = F.normalize(prod_emb, p=2, dim=-1)
        
        # Calculate Dot Product (which is now Cosine Similarity)
        return (user_emb * prod_emb).sum(dim=-1)

class GNNModel(nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        # Convert GNN to Heterogeneous GNN using the metadata
        self.encoder = to_hetero(self.encoder, metadata, aggr='sum')
        self.decoder = DotProductDecoder()

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # Ensure edge indices are LongTensor
        edge_index_dict = {k: v.to(torch.long).view(2, -1) for k, v in edge_index_dict.items()}
        
        # 1. Encode the graph to get node embeddings
        z_dict = self.encoder(x_dict, edge_index_dict)
        
        # 2. Decode using Dot Product
        return self.decoder(z_dict, edge_label_index)