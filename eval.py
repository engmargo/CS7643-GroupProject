import torch
import torch.nn.functional as F

def prepare_eval_data(data_maps, datasets_dict, split='test'):
    """
    Prepare evaluation data: map user_id/item_id to integer indices.
    
    Returns:
        user_ids: list of int user indices
        user2pos_items: dict {user_idx: set of item_idx}
    """
    dataset = datasets_dict[split]
    user_ids = []
    user2pos_items = {}

    for inter in dataset:
        # map user_id -> int
        uid_idx = data_maps['user2id'].get(inter['user_id'], 0)
        iid_idx = data_maps['item2id'].get(inter['item_id'], 0)

        user_ids.append(uid_idx)
        if uid_idx not in user2pos_items:
            user2pos_items[uid_idx] = set()
        user2pos_items[uid_idx].add(iid_idx)
    
    # remove duplicates if any
    user_ids = list(set(user_ids))
    return user_ids, user2pos_items


def eval_model(model, user_ids, user2pos_items, all_item_emb, topk=(5,10,20), device="cuda", batch_size=512):
    """
    Evaluate Two-Tower model: compute Recall@K, Precision@K, NDCG@K
    """
    model.eval()
    metrics = {f"Recall@{k}": 0.0 for k in topk}
    metrics.update({f"Precision@{k}": 0.0 for k in topk})
    metrics.update({f"NDCG@{k}": 0.0 for k in topk})
    
    all_item_emb = torch.tensor(all_item_emb, dtype=torch.float32).to(device)
    user_ids_tensor = torch.tensor(user_ids, device=device, dtype=torch.long).to(device)
    num_users = len(user_ids)
    
    with torch.no_grad():
        for start in range(0, num_users, batch_size):
            end = min(start + batch_size, num_users)
            batch_user_ids = user_ids_tensor[start:end]

            # encode user embeddings
            user_emb = model.encode_user(batch_user_ids)  # shape (B, D)
            user_emb = F.normalize(user_emb, dim=1)
            item_emb = F.normalize(all_item_emb, dim=1)

            # compute scores: (B, num_items)
            scores = torch.matmul(user_emb, item_emb.t())

            for k in topk:
                topk_scores, topk_idx = torch.topk(scores, k=k, dim=1)
                for i, u_idx in enumerate(batch_user_ids.tolist()):
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
                    dcg = sum([h / torch.log2(torch.tensor(idx + 2.0, dtype=torch.float)) for idx, h in enumerate(hits)])
                    idcg = sum([1.0 / torch.log2(torch.tensor(j + 2.0, dtype=torch.float)) for j in range(min(num_pos, k))])
                    if idcg > 0:
                        metrics[f"NDCG@{k}"] += (dcg / idcg).item()

    # average over users
    for key in metrics:
        metrics[key] /= num_users

    return metrics


def run_eval_pipeline(model, data_maps, all_item_emb, datasets_dict, split='test', device='cuda', batch_size=512, topk=(5,10,20)):
    """
    Full evaluation pipeline:
      - map user/item ids
      - prepare all_item_emb
      - compute Recall@K, Precision@K, NDCG@K
    """
    user_ids, user2pos_items = prepare_eval_data(data_maps, datasets_dict, split)
    
    metrics = eval_model(
        model=model,
        user_ids=user_ids,
        user2pos_items=user2pos_items,
        all_item_emb=all_item_emb,
        topk=topk,
        device=device,
        batch_size=batch_size
    )
    return metrics