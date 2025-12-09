import os
import json
import numpy as np 
import zipfile 
from datasets import Dataset, Features, Value
import pandas as pd 

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def unzip(zip_file_path, domain):
    temp_extract_dir = f'processed_data/'
    check_path(temp_extract_dir)

    print(f"unzip data from  {zip_file_path} to {temp_extract_dir}")
    
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if member.startswith(f'{domain}/') or member.startswith(f'processed/{domain}/'):
                    if member.startswith('processed/'):
                        target_name = member.replace('processed/', '', 1)
                    else:
                        target_name = member 

                    source = zip_ref.open(member)
                    target = os.path.join(temp_extract_dir, target_name)
                    
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    
                    if not target.endswith(os.sep):
                        with open(target, 'wb') as f:
                            f.write(source.read())

    except FileNotFoundError:
        print(f"Unfound: {zip_file_path}")
        return None, None, None
    except Exception as e:
        print(f"Unfound: {e}")

    return temp_extract_dir
    
def load_processed_data(temp_extract_dir, domain):
    domain_path = os.path.join(temp_extract_dir, domain)

    data_maps_path = os.path.join(domain_path, f'{domain}.data_maps')
    with open(data_maps_path, 'r') as f:
        data_maps = json.load(f)
    print(f"data_maps loaded.\n# of users : {len(data_maps['user2id']) - 1}\n # of items: {len(data_maps['item2id']) - 1}")

    # 5. item_features
    feature_file = [f for f in os.listdir(domain_path) if f.endswith('.feature')][0]
    item_features_path = os.path.join(domain_path, feature_file)
    
    FEATURE_DIM = 768 
    total_items = len(data_maps['item2id']) - 1 

    try:
        # use numpy.fromfile read binary-value file and reshape
        item_features = np.fromfile(item_features_path, dtype=np.float32).reshape(total_items, FEATURE_DIM)
        # add zero
        padding_vector = np.zeros((1, FEATURE_DIM), dtype=np.float32)
        item_features = np.concatenate([padding_vector, item_features], axis=0)
        print(f"shape of item feature: {item_features.shape}")
    except Exception as e:
        print(e)
        return None, None, None
    
    # 6. read iteratctions
    datasets_dict = {}
    
    inter_features = Features({
        'user_id': Value('string'),
        'item_id_list': Value('string'), # history
        'item_id': Value('string')      # target
    })

    for split in ['train', 'valid', 'test']:
        inter_file_path = os.path.join(domain_path, f'{domain}.{split}.inter')
        
        if os.path.exists(inter_file_path):
            df = pd.read_csv(
                inter_file_path, 
                sep='\t', 
                skiprows=[0], 
                names=['user_id', 'item_id_list', 'item_id']
            )
            datasets_dict[split] = Dataset.from_pandas(df, features=inter_features)
            print(f"{split}.inter loaded: {len(datasets_dict[split])}")
        else:
            print(f"File Not Found: {inter_file_path}. skip {split}。")
            
    return  data_maps, item_features,datasets_dict


def get_ft_by_inter(inter, data_maps, item_features):
    """
    Converts a single interaction record into user and item feature representations
    for Two-Tower model training.

    Args:
        inter: A single interaction record (e.g., from datasets['train'][i]).
               Expected fields: 'user_id', 'item_id_list', 'item_id'.
        data_maps: The dictionary containing 'user2id' and 'item2id'.
        item_features: The full item feature matrix (pre-computed embeddings).

    Returns:
        A tuple: (user_features, item_features).
    """
    
    # 1. User Features (Query Tower Input)
    
    # A. Get User ID Index (Static User Embedding)
    user_id = inter['user_id']
    user_idx = data_maps['user2id'].get(user_id, 0) # Use 0 for [PAD] if missing

    # B. Get History Sequence Indices (Dynamic Interest)
    # The history list is already cleaned and truncated, so we just convert IDs to indices.
    history_ids = inter['item_id_list'].split(' ') # Item IDs are stored as 'item_id_list' in your .inter file
    
    # Map item IDs to their numerical index (using 0 for [PAD] if an ID somehow got missed)
    history_indices = [data_maps['item2id'].get(item_id, 0) for item_id in history_ids if item_id]
    
    # Note: For training, the User Tower requires the user index AND the sequence of item indices.
    # We return the raw indices here; the DataLoader or training loop typically handles padding/tensor conversion.
    
    # In a typical PyTorch scenario, this might be returned as two components:
    user_features = np.array([user_idx]) # Static user ID index
    history_features = np.array(history_indices, dtype=np.int64) # Sequence of item indices

    # 2. Item Features (Candidate Tower Input - Target Item)

    # A. Get Target Item Index
    target_item_id = inter['item_id']
    target_item_idx = data_maps['item2id'].get(target_item_id, 0)

    # B. Fetch Item Feature Vector (PLM Embedding)
    # Since 'item_features' is the matrix of PLM embeddings (indexed 0=[PAD], 1=item1, ...),
    # we use the index to fetch the pre-computed feature vector.
    
    # Check bounds just in case, though the remapping should prevent index errors.
    if target_item_idx < item_features.shape[0]:
        item_ft = item_features[target_item_idx]
    else:
        # Fallback to PAD feature (index 0)
        item_ft = item_features[0] 
    
    return (user_features, history_features), item_ft