import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from twotower_config import *
from twotower_model import TwoTowerModel
from load_processed_data import unzip, load_processed_data, get_ft_by_inter
from twotower_dataset import TwoTowerTrainDataset


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def contrastive_loss(u: torch.Tensor, v: torch.Tensor, temperature: float = TEMPERATURE):
    # u, v: shape (B, D)
    # contrastive loss: info Noise-Contrastive Estimation (InfoNCE)
    logits = (u @ v.t()) / temperature
    labels = torch.arange(u.size(0), device=u.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def train(num_epochs: int = 10, temperature: float = TEMPERATURE, model_save_path: str = None, item_emb_save_path: str = None):
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # 1. load data
    DOMAIN = "Electronics"
    #ZIP_PATH = "processed_data.zip"
    #temp_dir = unzip(ZIP_PATH, DOMAIN)
    temp_dir = "/content/drive/MyDrive/CS7643-GroupProject-Colab/processed"
    data_maps, item_features_np, datasets_dict = load_processed_data(temp_dir, DOMAIN)

    num_users = len(data_maps["user2id"])  # includes index 0 (PAD)
    num_items = len(data_maps["item2id"])  # includes index 0 (PAD)
    FEATURE_DIM = item_features_np.shape[1]

    train_dataset = TwoTowerTrainDataset(
        hf_dataset=datasets_dict["train"],
        data_maps=data_maps,
        item_features=item_features_np,
        max_history_len=20,
        num_negatives=NUM_NEGATIVES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    # 2. initialize model
    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=EMBEDDING_DIM,
        item_feature_dim=FEATURE_DIM
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 3. train
    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}]", ncols=80)
        for batch in progress_bar:
            user_ids = batch["user_id"].to(device)
            history_item_ids = batch["history_item_ids"].to(device)
            item_ids = batch["item_id"].to(device)
            item_features = batch['item_feature'].to(device)

            neg_item_ids = batch["neg_item_ids"].to(device)
            neg_item_features = batch["neg_item_features"].to(device)

            # forward
            # positive item embedding
            user_emb, pos_item_emb = model(user_ids, history_item_ids, item_ids, item_features)
            # negative item embedding
            B, N, Fdim = neg_item_features.shape
            neg_item_ids_flat = neg_item_ids.view(-1)
            neg_item_features_flat = neg_item_features.view(B * N, Fdim)  # (B*N, F)
            neg_item_emb_flat = model.encode_item(neg_item_ids_flat, neg_item_features_flat)
            neg_item_emb = neg_item_emb_flat.view(B, N, -1)

            # calculate loss
            # positive sample scores (B,)
            pos_scores = (user_emb * pos_item_emb).sum(dim=-1) / temperature

            # negative sample scores (B,N)
            # neg_item_emb: (B, N, D)
            # user_emb.unsqueeze(-1): (B, D, 1)
            neg_scores = torch.bmm(neg_item_emb, user_emb.unsqueeze(-1)).squeeze(-1)  # (B, N)
            neg_scores = neg_scores / temperature

            #combined pos_scores and neg_scores into one logits (B, N+1)
            logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
            labels = torch.zeros(B, dtype=torch.long, device=device)

            # cross entropy loss when having negative samples
            loss = F.cross_entropy(logits, labels)

            #loss = contrastive_loss(u, v, temperature=temperature)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch}: avg loss = {avg_loss:.4f}")

    # 4. save model
    if model_save_path is None:
        model_save_path = os.path.join(PROCESSED_DIR, "two_tower_model.pt")
    if item_emb_save_path is None:
        item_emb_save_path = os.path.join(PROCESSED_DIR, "item_embeddings.pt")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    torch.save(model.state_dict(), model_save_path)
    print(f"Model parameters are saved to: {model_save_path}")

    # 5. calculate all item embeddings and save
    model.eval()
    with torch.no_grad():
        all_item_ids = torch.arange(num_items, device=device, dtype=torch.long)
        all_item_features = torch.from_numpy(item_features_np).to(device)
        all_item_emb = model.encode_item(all_item_ids, all_item_features)
        all_item_emb = all_item_emb.cpu()

    torch.save(
        {
            "item_embeddings": all_item_emb,
            "num_items": num_items,
            "embedding_dim": EMBEDDING_DIM,
        },
        item_emb_save_path,
    )
    print(f"All item embeddings are saved to: {item_emb_save_path}")


if __name__ == "__main__":
    train(num_epochs=NUM_EPOCHS)
