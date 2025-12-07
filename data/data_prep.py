
import argparse
import datasets
from datasets import get_dataset_config_names,load_dataset,Dataset
import os
import sys 
from data.generate_features import generate_ft
import yaml
import os
import sys 
import argparse

def load_config(config_path: str = 'config.yaml') -> dict:
    """
    Loads a YAML configuration file into a Python dictionary.
    """
    # Check if the file exists before attempting to open it
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    print(f"Loading configuration from {config_path}...")
    
    with open(config_path, 'r') as file:
        # Use safe_load to prevent arbitrary code execution, which is more secure
        config_data = yaml.safe_load(file)
        
    return config_data


dataset_id = "McAuley-Lab/Amazon-Reviews-2023"
configs_dataset = get_dataset_config_names(dataset_id,trust_remote_code=True)
configs = load_config()


def shrink_dataset(dataset,ratio):
    shrink_dataset = dataset.shuffle(seed=42).select(range(int(len(dataset) * ratio)))
    return shrink_dataset

def show_stats(dataset):
    ret = f'records:{len(dataset)}\n#_unique_users:{len(set(dataset['user_id']))}\n#_unique_items:{len(set(dataset['parent_asin']))}\n'
    print(ret)
    
    
def load_data(cat:str = 'Electronics'):
    split_interactions_config = f'5core_timestamp_w_his_{cat}' if configs.history_ft_included else f'5core_timestamp_{cat}'
    item_config = f'raw_meta_{cat}'
    reviews_config = f'raw_review_{cat}'
    if split_interactions_config not in configs_dataset or item_config not in configs_dataset or reviews_config not in configs_dataset:
        raise ValueError()
    
    split_ids = load_dataset(dataset_id, split_interactions_config , trust_remote_code=True)
    for key,dt in split_ids.items():
        print(f'dataset - {key}:')
        show_stats(dt)
    
    items = load_dataset(dataset_id, item_config,split = 'full',trust_remote_code=True)
    # reviews = load_dataset(dataset_id, reviews_config, trust_remote_code=True)
    
    return split_ids,items


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='Electronics')
    parser.add_argument('--max_his_len', type=int, default=50)
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='processed/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--plm', type=str, default='hyp1231/blair-roberta-base')
    parser.add_argument('--batch_size', type=int, default=16)
    return parser.parse_args()

def data_prep_main():
    split_ids,items = load_data()
    shrink_split_ids = {}
    for key,dt in split_ids.items():
        shrink_dt = shrink_dataset(dt,configs['shrink_ratio'][key])
        shrink_split_ids[key] = shrink_dt
        dataset_output_path = os.path.join(args.output_dir, f'{args.domain}.{key}.hf_dataset')
        shrink_dt.save_to_dist(dataset_output_path)
        show_stats(shrink_dataset)
    del split_ids
    
    args = parse_args()
    
    shrink_items = shrink_dataset(items,configs['shrink_ratio']['items'])
    dataset_output_path = os.path.join(args.output_dir, f'{args.domain}.items.hf_dataset')
    shrink_items.save_to_dist(dataset_output_path)
        
    # Use the save_to_disk method
    generate_ft(items,shrink_split_ids,args)
    
    
    



    

    
    
    
    
    
    

    
    
