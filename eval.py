import torch
import torch.nn.functional as F

def eval_model(model, user_ids, user2pos_items, all_item_emb, topk=(5,10,20), device="cuda", batch_size=512):
    """
    Evaluate a recommender model and compute Recall@K, Precision@K, NDCG@K.

    Args:
        model: The recommender model with methods `encode_user` and `encode_item`
        user_ids: list of user ids to evaluate
        user2pos_items: dict, {user_id: set of positive item ids for evaluation}
        all_item_emb: Tensor, (num_items, D)
        topk: tuple of k values to compute metrics
        device: cuda or cpu
        batch_size: batch size for evaluation

    Returns:
        metrics: dict with keys like 'Recall@5', 'Precision@5', 'NDCG@5', ...
    """
    model.eval()
    metrics = {f"Recall@{k}": 0.0 for k in topk}
    metrics.update({f"Precision@{k}": 0.0 for k in topk})
    metrics.update({f"NDCG@{k}": 0.0 for k in topk})
    
    all_item_emb = all_item_emb.to(device)
    user_ids_tensor = torch.tensor(user_ids, device=device, dtype=torch.long)
    num_users = len(user_ids)
    
    with torch.no_grad():
        for start in range(0, num_users, batch_size):
            end = min(start + batch_size, num_users)
            batch_user_ids = user_ids_tensor[start:end]
            
            # 1) encode user embeddings
            user_emb = model.encode_user(batch_user_ids)  # (B, D)
            user_emb = F.normalize(user_emb, dim=1)
            item_emb = F.normalize(all_item_emb, dim=1)
            
            # 2) compute similarity
            scores = torch.matmul(user_emb, item_emb.t())  # (B, num_items)
            
            for k in topk:
                # 3) get top-k recommended items
                topk_scores, topk_idx = torch.topk(scores, k=k, dim=1)
                
                for i, u_id in enumerate(batch_user_ids.tolist()):
                    pos_items = user2pos_items.get(u_id, set())
                    if not pos_items:
                        continue
                    
                    # 4) compute hits
                    hits = [1 if item in pos_items else 0 for item in topk_idx[i].tolist()]
                    num_pos = len(pos_items)
                    
                    # Recall@K
                    metrics[f"Recall@{k}"] += sum(hits) / num_pos
                    
                    # Precision@K
                    metrics[f"Precision@{k}"] += sum(hits) / k
                    
                    # NDCG@K
                    dcg = sum([h / torch.log2(torch.tensor(idx+2.0, dtype=torch.float)) for idx, h in enumerate(hits)])
                    idcg = sum([1.0 / torch.log2(torch.tensor(j+2.0, dtype=torch.float)) for j in range(min(num_pos, k))])
                    if idcg > 0:
                        metrics[f"NDCG@{k}"] += (dcg / idcg).item()
    
    # 5) average over all users
    for key in metrics:
        metrics[key] /= num_users
    
    return metrics