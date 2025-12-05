import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from twotower_config import *
from dataset import get_train_loader
from twotower_model import TwoTowerModel


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Mac M1/M2
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def contrastive_loss(u: torch.Tensor, v: torch.Tensor, temperature: float = 1.0):
    # u, v: shape (B, D)
    # contrastive loss: info Noise-Contrastive Estimation (InfoNCE)
    logits = (u @ v.t()) / temperature
    labels = torch.arange(u.size(0), device=u.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def train(num_epochs: int = 10, temperature: float = 1.0, model_save_path: str = None, item_emb_save_path: str = None):
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # 1. load data
    train_loader, num_users, num_items = get_train_loader(BATCH_SIZE)
    print(f"num_users = {num_users}, num_items = {num_items}")

    # 2. initialize model
    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=EMBEDDING_DIM,
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
            item_ids = batch["item_id"].to(device)
            item_features = batch['item_feature'].to(device)

            # forward
            u, v = model(user_ids, item_ids, item_features)

            # calculate loss
            loss = contrastive_loss(u, v, temperature=temperature)

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

    # calculate all item embeddings and save
    model.eval()
    with torch.no_grad():
        all_item_ids = torch.arange(num_items, device=device, dtype=torch.long)
        item_feature_path = "data/processed/item_features.npy"
        all_item_features = torch.from_numpy(np.load(item_feature_path)).to(device)
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
    train()
