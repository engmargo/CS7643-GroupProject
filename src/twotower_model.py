import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, item_feature_dim):
        super().__init__()

        # user id embedding
        self.user_id_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        # user MLP
        self.user_mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

        # item id embedding
        self.item_id_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        # item feature mlp
        self.item_feature_mlp = nn.Sequential(
            nn.Linear(item_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        # combine item id embedding and item feature mlp
        self.item_tower = nn.Linear(2*embedding_dim,embedding_dim)

    def encode_user(self, user_ids, history_indices):
        #print("user_ids shape:", user_ids.shape)
        #print("history_indices shape:", history_indices.shape)
        user_id_emb = self.user_id_embedding(user_ids) # (B, D)
        history_emb = self.item_id_embedding(history_indices) # (B, L, D)
        history_pool = history_emb.mean(dim=1)  # (B, D)
        concat = torch.cat([user_id_emb, history_pool], dim=-1) #(B, 2D)
        u = self.user_mlp(concat)
        return F.normalize(u, dim=-1)

    def encode_item(self, item_ids, item_features):
        #print("item_ids shape:", item_ids.shape)
        #print("item_features shape:", item_features.shape)
        v_id = self.item_id_embedding(item_ids)
        v_feature = self.item_feature_mlp(item_features) # MLP
        concat = torch.cat([v_id,v_feature],dim=-1)
        v = self.item_tower(concat)
        return F.normalize(v, dim=-1)

    def forward(self, user_ids, history_indices, item_ids, item_features):
        u = self.encode_user(user_ids, history_indices)
        v = self.encode_item(item_ids, item_features)
        return u, v
