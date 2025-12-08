from torch.utils.data import Dataset
import torch
import numpy as np
from load_processed_data import unzip, load_processed_data, get_ft_by_inter

class TwoTowerTrainDataset(Dataset):
    def __init__(self, hf_dataset, data_maps, item_features, max_history_len: int = 20):
        self.hf_dataset = hf_dataset           # HuggingFace Dataset: train / valid / test
        self.data_maps = data_maps
        self.item_features = item_features     # numpy array, shape (num_items+1, 768)
        self.max_history_len = max_history_len

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # get one interaction
        inter = self.hf_dataset[idx]

        # get (user_features, history_features), item_ft
        (user_features, history_features), item_ft = get_ft_by_inter(
            inter, self.data_maps, self.item_features
        )

        # 1. user index
        user_idx = int(user_features[0])

        # 2. history item indices: shape 1D np.array，fixed length
        hist = history_features.astype(np.int64)

        # max_history_len
        if len(hist) > self.max_history_len:
            hist = hist[-self.max_history_len:]

        # left padding 0
        if len(hist) < self.max_history_len:
            pad_len = self.max_history_len - len(hist)
            pad = np.zeros(pad_len, dtype=np.int64)
            hist = np.concatenate([pad, hist], axis=0)  # (L,)

        # 3. target item index
        target_item_id = inter["item_id"]
        item_idx = self.data_maps["item2id"].get(target_item_id, 0)

        return {
            "user_id": torch.tensor(user_idx, dtype=torch.long),  # ()
            "history_item_ids": torch.tensor(hist, dtype=torch.long),  # (L,)
            "item_id": torch.tensor(item_idx, dtype=torch.long),  # ()
            "item_feature": torch.tensor(item_ft, dtype=torch.float32),  # (FEATURE_DIM,)
        }

