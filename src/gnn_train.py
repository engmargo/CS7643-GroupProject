import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import yaml
import os
import sys
import numpy as np
from tqdm import tqdm
from gnn_model import GNNModel

# Add parent directory and 'src' subdirectory to sys.path to find load_processed_data.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'src'))

from load_processed_data import load_processed_data

def load_config(path="configs/gnn_config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_graph_data(config):
    """
    Uses the shared 'load_processed_data' utility to load:
    1. Data Maps (User/Item ID mappings)
    2. Item Features (Pre-computed RoBERTa embeddings)
    3. Datasets (HuggingFace datasets for Train/Valid/Test)
    """
    processed_dir = config['dataset']['processed_dir']
    domain = config['dataset']['name']
    
    print(f"Loading processed data for {domain} from {processed_dir}...")
    
    data_maps, item_features_np, datasets_dict = load_processed_data(processed_dir, domain)

    num_users = len(data_maps['user2id'])
    num_items = len(data_maps['item2id'])
    print(f"Stats: {num_users} Users, {num_items} Items")

    # Initialize HeteroData Object
    data = HeteroData()

    # --- Node Features ---
    data['Product'].x = torch.from_numpy(item_features_np).float()
    data['User'].x = torch.randn(num_users, config['dataset']['num_user_features'])

    # --- Build Edges ---
    print("Constructing Graph Edges...")
    
    mp_src = [] # User IDs
    mp_dst = [] # Item IDs

    split_edge_indices = {} # To store (user, target) pairs for supervision

    for mode in ['train', 'valid', 'test']:
        if mode not in datasets_dict: continue
        
        print(f"Processing {mode} set...")
        ds = datasets_dict[mode]
        
        u_list = ds['user_id']
        i_list = ds['item_id']
        hist_list = ds['item_id_list']

        # Map to integers
        u_indices = [data_maps['user2id'].get(str(u), 0) for u in u_list]
        i_indices = [data_maps['item2id'].get(str(i), 0) for i in i_list]
        
        # Store supervision edges
        split_edge_indices[mode] = torch.tensor([u_indices, i_indices], dtype=torch.long)

        # Extract Message Passing Edges from History
        for u_idx, h_str in zip(u_indices, tqdm(hist_list, desc=f"Parsing {mode} history")):
            if not h_str: continue
            hist_items = str(h_str).split(' ')
            for h_asin in hist_items:
                if h_asin in data_maps['item2id']:
                    mp_src.append(u_idx)
                    mp_dst.append(data_maps['item2id'][h_asin])

    # Construct the global graph topology
    edge_index = torch.tensor([mp_src, mp_dst], dtype=torch.long)
    edge_index = torch.unique(edge_index, dim=1)
    
    data['User', 'reviews', 'Product'].edge_index = edge_index
    
    # Make Undirected
    data = T.ToUndirected()(data)
    
    return data, split_edge_indices

def train_epoch(model, optimizer, data, train_edge_index, batch_size):
    """
    Performs one epoch of training using Mini-Batch SGD.
    """
    model.train()
    
    num_edges = train_edge_index.size(1)
    
    # Shuffle edges for this epoch
    perm = torch.randperm(num_edges, device=train_edge_index.device)
    train_edges = train_edge_index[:, perm]
    
    total_loss = 0
    num_batches = 0
    
    # Iterate in batches
    for i in range(0, num_edges, batch_size):
        optimizer.zero_grad()
        
        # 1. Get Batch Positive Edges
        batch_pos_edges = train_edges[:, i : i + batch_size]
        current_batch_size = batch_pos_edges.size(1)
        
        # 2. Negative Sampling (Randomly sample products for this batch)
        # We sample random items from the Product node set
        neg_items = torch.randint(0, data['Product'].num_nodes, (current_batch_size,), device=train_edge_index.device)
        
        # Construct negative pairs: (same_user, random_item)
        batch_neg_edges = torch.stack([batch_pos_edges[0], neg_items], dim=0)
        
        # 3. Forward Pass
        # Note: We pass the FULL graph structure (data.edge_index_dict) for message passing context,
        # but we only predict/compute loss on the BATCH edges.
        pos_pred = model(data.x_dict, data.edge_index_dict, batch_pos_edges)
        neg_pred = model(data.x_dict, data.edge_index_dict, batch_neg_edges)
        
        # 4. Loss Calculation (MSE: Pos -> 1.0, Neg -> 0.0)
        all_pred = torch.cat([pos_pred, neg_pred])
        all_label = torch.cat([torch.ones(current_batch_size, device=pos_pred.device), 
                               torch.zeros(current_batch_size, device=neg_pred.device)])
        
        loss = F.mse_loss(all_pred, all_label)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / num_batches

@torch.no_grad()
def evaluate(model, data, edge_index):
    model.eval()
    # For evaluation, we can process all edges at once if memory allows, 
    # otherwise we should batch this too. Given standard validation sizes, this usually fits.
    pred = model(data.x_dict, data.edge_index_dict, edge_index)
    return pred.mean().item()

def main():
    config = load_config()
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    set_seed()

    # Load Data
    try:
        data, split_indices = build_graph_data(config)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    data = data.to(device)
    train_idx = split_indices['train'].to(device)
    valid_idx = split_indices['valid'].to(device)

    model = GNNModel(
        hidden_channels=config['model']['hidden_channels'],
        metadata=data.metadata()
    ).to(device)

    # Lazy Init
    with torch.no_grad():
        model(data.x_dict, data.edge_index_dict, train_idx[:, :2])

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['train']['learning_rate'], 
        weight_decay=config['train']['weight_decay']
    )
    
    batch_size = config['train']['batch_size']
    print(f"Starting training with Batch Size: {batch_size}...")
    
    for epoch in range(1, config['train']['epochs'] + 1):
        loss = train_epoch(model, optimizer, data, train_idx, batch_size)
        
        if epoch % 1 == 0:
            val_score = evaluate(model, data, valid_idx)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Pos Score: {val_score:.4f}')

if __name__ == "__main__":
    main()