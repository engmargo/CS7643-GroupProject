from torch.utils.data import Dataset
import torch
from load_processed_data import unzip, load_processed_data, get_ft_by_inter

class TwoTowerTrainDataset(Dataset):
    def __init__(self, hf_dataset, data_maps, item_features):
        self.hf_dataset = hf_dataset           # HuggingFace Dataset: train / valid / test
        self.data_maps = data_maps
        self.item_features = item_features     # numpy array, shape (num_items+1, 768)

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # 1. get one interaction
        inter = self.hf_dataset[idx]

        # 2. get (user_features, history_features), item_ft
        (user_idx_arr, history_indices), item_ft = get_ft_by_inter(
            inter, self.data_maps, self.item_features
        )
        # user_idx_arr: np.array([user_idx])
        # history_indices: np.array([...]) ignore history
        # item_ft: np.array(768,)

        user_idx = user_idx_arr[0]   # 取出 int
        # ignore history_indices

        # 3. keep item_id index
        target_item_id = inter["item_id"]
        item_idx = self.data_maps["item2id"].get(target_item_id, 0)

        return {
            "user_id": torch.tensor(user_idx, dtype=torch.long),
            "item_id": torch.tensor(item_idx, dtype=torch.long),
            "item_feature": torch.tensor(item_ft, dtype=torch.float32),
            # "history": torch.tensor(history_indices, dtype=torch.long),
        }

