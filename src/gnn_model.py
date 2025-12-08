import torch
import torch.nn as nn
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

class EdgeDecoder(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        # Concatenate user and product embeddings to predict edge existence
        z = torch.cat([z_dict['User'][row], z_dict['Product'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

class GNNModel(nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        # Convert GNN to Heterogeneous GNN using the metadata
        self.encoder = to_hetero(self.encoder, metadata, aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # Ensure edge indices are LongTensor
        edge_index_dict = {k: v.to(torch.long).view(2, -1) for k, v in edge_index_dict.items()}
        
        # 1. Encode the graph to get node embeddings
        z_dict = self.encoder(x_dict, edge_index_dict)
        
        # 2. Decode the specific edges (training or testing pairs)
        return self.decoder(z_dict, edge_label_index)
        