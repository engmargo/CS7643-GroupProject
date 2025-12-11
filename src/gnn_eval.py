import torch
import torch.nn as nn
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'src'))

from eval import eval_model
from gnn_train import load_config, build_graph_data, GNNModel, load_processed_data

class GNNAdapter(nn.Module):
    """
    Wraps the pre-computed user embeddings so eval_model can call encode_user.
    Inherits from nn.Module to support methods like .eval() and .to().
    """
    def __init__(self, user_embeddings):
        super().__init__()
        # Register buffer so it moves devices with the model if needed
        self.register_buffer("user_embeddings", user_embeddings)

    def encode_user(self, user_ids):
        # user_ids is a tensor of indices. We simply slice our pre-computed embeddings.
        return self.user_embeddings[user_ids]

def evaluate_gnn():
    # --- Setup ---
    config = load_config()
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Data & Model ---
    # We reuse build_graph_data to get the graph and maps
    data, split_indices = build_graph_data(config)
    data = data.to(device)
    
    # Load raw data maps to convert test set IDs to Indices
    processed_dir = config['dataset']['processed_dir']
    domain = config['dataset']['name']
    data_maps, _, datasets_dict = load_processed_data(processed_dir, domain)
    
    # Initialize Model
    model = GNNModel(
        hidden_channels=config['model']['hidden_channels'],
        metadata=data.metadata()
    ).to(device)

    # --- LOAD TRAINED WEIGHTS (Added) ---
    try:
        model.load_state_dict(torch.load("gnn_model.pt", map_location=device))
        print("Successfully loaded trained model weights from 'gnn_model.pt'.")
    except FileNotFoundError:
        print("WARNING: 'gnn_model.pt' not found. Using random weights (metrics will be ~0).")
        print("Please run gnn_train.py first.")
    
    # --- Generate Embeddings ---
    print("Generating GNN node embeddings...")
    model.eval()
    with torch.no_grad():
        # Run the encoder on the full graph to get embeddings for ALL users and items
        z_dict = model.encoder(data.x_dict, data.edge_index_dict)
        
        # Extract embeddings
        user_emb_all = z_dict['User']      # Shape: [num_users, hidden_dim]
        item_emb_all = z_dict['Product']   # Shape: [num_items, hidden_dim]

    # --- Prepare Evaluation Data ---
    print("Preparing Ground Truth (User -> Positive Items)...")
    test_dataset = datasets_dict['test']
    user2pos_items = {}
    
    # We need to list all user IDs we want to evaluate
    eval_user_ids = []

    # Iterate over test dataset and map raw IDs to Model Indices
    # Note: Using tqdm here is recommended if dataset is large
    from tqdm import tqdm
    for i in tqdm(range(len(test_dataset)), desc="Mapping IDs"):
        raw_uid = test_dataset[i]['user_id']
        raw_iid = test_dataset[i]['item_id']

        # Map to index
        u_idx = data_maps['user2id'].get(str(raw_uid))
        i_idx = data_maps['item2id'].get(str(raw_iid))

        # Skip if user/item not in training graph
        if u_idx is not None and i_idx is not None:
            if u_idx not in user2pos_items:
                user2pos_items[u_idx] = set()
                eval_user_ids.append(u_idx)
            user2pos_items[u_idx].add(i_idx)

    # --- Run Evaluation ---
    print(f"Evaluating on {len(eval_user_ids)} users...")
    
    # Create the adapter
    gnn_adapter = GNNAdapter(user_emb_all)

    # Call the provided eval_model function
    metrics = eval_model(
        model=gnn_adapter,           # Pass our adapter instead of TwoTowerModel
        user_ids=eval_user_ids,      # List of user indices to test
        user2pos_items=user2pos_items,
        all_item_emb=item_emb_all,   # Pass the product embeddings from GNN
        topk=config.get('eval', {}).get('topk_list', [10, 20, 50]), 
        device=device,
        batch_size=config['train']['batch_size']
    )

    # --- Print Results ---
    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    evaluate_gnn()