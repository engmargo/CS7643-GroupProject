import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from twotower_config import *
from twotower_model import TwoTowerModel
from load_processed_data import unzip, load_processed_data, get_ft_by_inter
from twotower_dataset import TwoTowerTrainDataset


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
    """
    根据 data_maps 和 item_features 的形状创建 TwoTowerModel，
    加载训练好的参数和保存好的 item_embeddings。
    """
    num_users = len(data_maps["user2id"])
    num_items = len(data_maps["item2id"])
    item_feature_dim = item_features.shape[1]

    # load trained model and emb
    if model_path is None:
        model_path = os.path.join(PROCESSED_DIR, "two_tower_model.pt")
    if item_emb_path is None:
        item_emb_path = os.path.join(PROCESSED_DIR, "item_embeddings.pt")

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
    split: str = "valid",
    batch_size: int = 128,
    max_history_len: int = 20,
):
    device = get_device()
    print(f"Using device: {device}")

    # 1. load dataset
    DOMAIN = "Electronics"
    temp_dir = "/content/drive/MyDrive/CS7643-GroupProject-Colab/processed"
    data_maps, item_features_np, datasets_dict = load_processed_data(temp_dir, DOMAIN)

    eval_dataset = TwoTowerTrainDataset(
        hf_dataset=datasets_dict[split],
        data_maps=data_maps,
        item_features=item_features_np,
        max_history_len=20,
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

    # 3. calculate Recall@K / MRR@K
    topk_list = sorted(TOPK_LIST)
    max_k = max(topk_list)

    metrics = {f"Recall@{k}": 0.0 for k in topk_list}
    metrics.update({f"MRR@{k}": 0.0 for k in topk_list})

    num_samples = 0

    with torch.no_grad():
        for batch in eval_loader:
            user_ids = batch["user_id"].to(device)
            history_item_ids = batch["history_item_ids"].to(device)
            target_item_ids = batch["item_id"].to(device)

            # user embeddings
            user_emb = model.encode_user(user_ids, history_item_ids)  # (B, D)
            user_emb = F.normalize(user_emb, dim=1)

            # similarity scores to all items
            scores = torch.matmul(user_emb, all_item_emb.t())     # (B, num_items)

            topk_scores, topk_indices = torch.topk(scores, k=max_k, dim=1)  # (B, max_k)

            batch_size_actual = user_ids.size(0)
            num_samples += batch_size_actual

            for i in range(batch_size_actual):
                target = target_item_ids[i].item()
                ranked = topk_indices[i].tolist()  # 长度 max_k

                for k in topk_list:
                    topk_items = ranked[:k]

                    # hit / recall
                    hit = 1.0 if target in topk_items else 0.0
                    metrics[f"Recall@{k}"] += hit  # Recall==hit

                    # MRR@K
                    if hit:
                        rank = topk_items.index(target) + 1  # 1-based
                        metrics[f"MRR@{k}"] += 1.0 / rank

    # 5. mean
    for key in metrics:
        metrics[key] /= float(num_samples)

    print(f"\n[Two-Tower] Evaluation on split = {split}")
    for k in topk_list:
        r = metrics[f"Recall@{k}"]
        m = metrics[f"MRR@{k}"]
        print(f"  Recall@{k}: {r:.4f} | MRR@{k}: {m:.4f}")

    return metrics


if __name__ == "__main__":
    eval_twotower(split="test")
