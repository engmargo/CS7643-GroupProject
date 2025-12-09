import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from twotower_config import *
from twotower_model import TwoTowerModel
from load_processed_data import unzip, load_processed_data, get_ft_by_inter
from twotower_dataset import TwoTowerTrainDataset
import argparse
import time 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='test')
    return parser.parse_args()

args = parse_args()

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_model_and_items(
    data_maps,
    item_features: np.ndarray,
    device: torch.device,
    model_path: str = None,
    item_emb_path: str = None,
):
    num_users = len(data_maps["user2id"])
    num_items = len(data_maps["item2id"])
    item_feature_dim = item_features.shape[1]

    # load trained model and emb
    if model_path is None:
        model_path = os.path.join(PROCESSED_DIR, "two_tower_model.pt")
    if item_emb_path is None:
        item_emb_path = os.path.join(PROCESSED_DIR, "two_tower_item_embeddings.pt")

    # initialize model
    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=EMBEDDING_DIM,
        item_feature_dim=item_feature_dim,
    )

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # load item embedding
    item_ckpt = torch.load(item_emb_path, map_location=device)
    all_item_emb = item_ckpt["item_embeddings"].to(device)  # (num_items, D)
    all_item_emb = F.normalize(all_item_emb, dim=1)

    return model, all_item_emb


def eval_twotower(
    split: str = 'test',
    batch_size: int = 128,
    max_history_len: int = 20,
    topk = (10,20,50)
):
    device = get_device()
    print(f"Using device: {device}")

    # 1. load dataset
    DOMAIN = "Electronics"
    temp_dir = PROCESSED_DIR
    data_maps, item_features_np, datasets_dict = load_processed_data(temp_dir, DOMAIN)

    user2pos_items = {}
    for row in datasets_dict[split]:
        u = data_maps["user2id"][row["user_id"]]
        i = data_maps["item2id"][row["item_id"]]
        user2pos_items.setdefault(u, set()).add(i)
        
    eval_dataset = TwoTowerTrainDataset(
        hf_dataset=datasets_dict[split],
        data_maps=data_maps,
        item_features=item_features_np,
        max_history_len=max_history_len,
        num_negatives=NUM_NEGATIVES
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    # 2. load trained model & item embeddings
    model, all_item_emb = load_model_and_items(
        data_maps=data_maps,
        item_features=item_features_np,
        device=device,
    )

    # 3. calculate metrics
    metrics = {f"Recall@{k}": 0.0 for k in topk}
    metrics.update({f"Precision@{k}": 0.0 for k in topk})
    metrics.update({f"NDCG@{k}": 0.0 for k in topk})

    num_samples = 0

    with torch.no_grad():
        times = []
        for batch in eval_loader:
            user_ids = batch["user_id"].to(device)
            history_item_ids = batch["history_item_ids"].to(device)
            target_item_ids = batch["item_id"].to(device)

            t0 = time.time()
            # user embeddings
            user_emb = model.encode_user(user_ids, history_item_ids)  # (B, D)
            user_emb = F.normalize(user_emb, dim=1)

            # similarity scores to all items
            scores = torch.matmul(user_emb, all_item_emb.t())     # (B, num_items)

            for k in topk:
                topk_scores, topk_idx = torch.topk(scores, k=k, dim=1)
                for i, u_idx in enumerate(user_ids.tolist()):
                    pos_items = user2pos_items.get(u_idx, set())
                    if not pos_items:
                        continue
                    hits = [1 if item in pos_items else 0 for item in topk_idx[i].tolist()]
                    num_pos = len(pos_items)
                    # Recall@K
                    metrics[f"Recall@{k}"] += sum(hits) / num_pos
                    # Precision@K
                    metrics[f"Precision@{k}"] += sum(hits) / k
                    # NDCG@K
                    dcg = sum([h / torch.log2(torch.tensor(idx + 2.0)) for idx, h in enumerate(hits)])
                    idcg = sum([1.0 / torch.log2(torch.tensor(j + 2.0)) for j in range(min(num_pos, k))])
                    if idcg > 0:
                        metrics[f"NDCG@{k}"] += (dcg / idcg).item()

            num_samples += len(user_ids)
            t1 = time.time()
            times.append(t1 - t0)

    # 5. mean
    for key in metrics:
        metrics[key] /= num_samples

    print(f"[{split}]Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)
    print('Avg Inference Latency:',(sum(times) / len(times)) / batch_size*1000)


if __name__ == "__main__":
    eval_twotower(args.split)
