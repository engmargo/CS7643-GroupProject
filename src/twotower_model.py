import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP
        self.user_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

        self.item_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def encode_user(self, user_ids):
        u = self.user_embedding(user_ids)
        u = self.user_mlp(u) # MLP
        return F.normalize(u, dim=-1)

    def encode_item(self, item_ids):
        v = self.item_embedding(item_ids)
        v = self.item_mlp(v) # MLP
        return F.normalize(v, dim=-1)

    def forward(self, user_ids, pos_item_ids):
        u = self.encode_user(user_ids)
        v = self.encode_item(pos_item_ids)
        return u, v
