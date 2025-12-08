import yaml
import numpy as np
import scipy.sparse as sp
import datasets
from datasets import load_dataset, DatasetDict, Dataset
import os
import pickle
import json

class AmazonGraphLoader:
    def __init__(self, config_filename="lightgnn_config.yaml",config_path = None):
        if config_path is None:
            ROOT = os.path.dirname(os.path.dirname(__file__))
            self.config_path = os.path.join(ROOT, "configs", config_filename)
        else:
            self.config_path = config_path

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.user_map = {} # user_str -> user_int
        self.item_map = {} # asin_str -> item_int
        self.n_users = 0
        self.n_items = 0

        self.dataset_dict = None
    
    def load_raw_data(self):
        data_cfg = self.config['data']
        path = data_cfg['path']
        file_type = data_cfg['file_type']
        print(f"Loading local dataset from: {path}...")

        if file_type == 'json':
            with open(path, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
                raw_dataset = Dataset.from_dict(data_dict)
                raw_dataset = DatasetDict({'train': raw_dataset})
        
        else:
            raw_dataset = load_dataset(data_cfg['file_type'], data_files=path)

        self.dataset_dict = self.perform_shuffle_and_split(raw_dataset)
        print(f"Found splits: {list(self.dataset_dict.keys())}")
    
    def perform_shuffle_and_split(self, raw_dataset):
        
        needed_splits = set(self.config['data']['split'])
        existing_splits = set(raw_dataset.keys())
        
        if needed_splits.issubset(existing_splits):
            return raw_dataset

        ratios = self.config['data'].get('split_ratios')
        train_ratio, test_ratio, val_ratio = ratios
        
        total = sum(ratios)
        test_size = test_ratio / total
        val_size = val_ratio / total
        

        full_data = raw_dataset['train']
        train_testval = full_data.train_test_split(test_size=(test_size + val_size), seed=2025)
        
        val_relative_ratio = val_size / (test_size + val_size)
        
        test_val = train_testval['test'].train_test_split(test_size=val_relative_ratio, seed=2025)
        
        final_dict = DatasetDict({
            'train': train_testval['train'],
            'test': test_val['train'],      # test part
            'validation': test_val['test']  # validation
        })
        
        for k, v in final_dict.items():
            print(f"  -> {k}: {len(v)} examples")
            
        return final_dict

    def build_global_mappings(self):
        
        all_users = []
        all_items = []
                
        for split_name in self.dataset_dict.keys():             
            data = self.dataset_dict[split_name]
            all_users.append(np.array(data['user_id']))
            all_items.append(np.array(data['parent_asin']))
            

        combined_users = np.concatenate(all_users)
        combined_items = np.concatenate(all_items)
        unique_users = np.unique(combined_users)
        unique_items = np.unique(combined_items)
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        
        self.user_map = {uid: i for i, uid in enumerate(unique_users)}
        self.item_map = {asin: i for i, asin in enumerate(unique_items)}
        
        print(f"Global Stats: {self.n_users} Users, {self.n_items} Items")

    def process_split(self, split_name):

        print(f"Processing split: {split_name}...")
        
        if split_name not in self.dataset_dict:
            return None
            
        data = self.dataset_dict[split_name]
        raw_users = data['user_id']
        raw_items = data['parent_asin']
        raw_ratings = data['rating']
        
        min_rating = self.config['data'].get('min_rating', 0)
        
        u_indices = []
        i_indices = []
        
        for u, i, r in zip(raw_users, raw_items, raw_ratings):
            if float(r) >= min_rating:
                if u in self.user_map and i in self.item_map:
                    u_indices.append(self.user_map[u])
                    i_indices.append(self.item_map[i])
        
        # Interaction Matrix
        n_interactions = len(u_indices)
        data_ones = np.ones(n_interactions, dtype=np.float32)
        
        # Shape of all matrix:(Global_N_Users, Global_N_Items)
        mat = sp.coo_matrix(
            (data_ones, (u_indices, i_indices)), 
            shape=(self.n_users, self.n_items)
        )
        
        print(f"  - {split_name}: {n_interactions} interactions")
        return mat

    def build_train_adjacency(self, train_mat):

        print("Building Adjacency Matrix for TRAIN data...")
        
        R = train_mat
        R_csr = R.tocsr()
        
        adj_mat = sp.bmat([
            [None, R_csr],
            [R_csr.T, None]
        ], format='csr')
        
        return adj_mat

    def save_all(self):
        save_path = self.config['data']['save_path']
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        with open(os.path.join(save_path, "user_map.pkl"), "wb") as f:
            pickle.dump(self.user_map, f)
        with open(os.path.join(save_path, "item_map.pkl"), "wb") as f:
            pickle.dump(self.item_map, f)

        splits_map = {
            'train': 'TrnMat.pkl',
            'test': 'TstMat.pkl',
            'validation': 'ValMat.pkl'
        }
        
        for split_name,filename in splits_map.items():
            mat = self.process_split(split_name)
            if mat is None: continue
            
            if split_name == "train": 
                with open(os.path.join(save_path, "TrnMat.pkl"), "wb") as f:
                    pickle.dump(mat, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                adj_mat = self.build_train_adjacency(mat)
                sp.save_npz(os.path.join(save_path, "adj_mat.npz"), adj_mat)
                
            elif split_name == "test":
                with open(os.path.join(save_path, "TstMat.pkl"), "wb") as f:
                    pickle.dump(mat, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
            elif split_name == "validation":
                with open(os.path.join(save_path, "ValMat.pkl"), "wb") as f:
                    pickle.dump(mat, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"All files saved to {save_path}")


if __name__ == "__main__":
    loader = AmazonGraphLoader()
    loader.load_raw_data()       
    loader.build_global_mappings() 
    loader.save_all()            