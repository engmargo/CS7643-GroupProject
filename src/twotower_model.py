import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, item_feature_dim):
        super().__init__()

        # user id embedding
        self.user_id_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)

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

    def encode_user(self, user_ids):
        u = self.user_id_embedding(user_ids)
        return F.normalize(u, dim=-1)

    def encode_item(self, item_ids, item_features):
        v_id = self.item_id_embedding(item_ids)
        v_feature = self.item_feature_mlp(item_features) # MLP
        concat = torch.cat([v_id,v_feature],dim=-1)
        v = self.item_tower(concat)
        return F.normalize(v, dim=-1)

    def forward(self, user_ids, item_ids, item_features):
        u = self.encode_user(user_ids)
        v = self.encode_item(item_ids, item_features)
        return u, v
