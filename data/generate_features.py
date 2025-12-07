import os
import torch
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
import numpy as np 
import re
import html
import gc

def filter_items_wo_metadata(example, item2meta):
    parent_asin = example.get('parent_asin')
    if not isinstance(parent_asin, str) or not parent_asin:
        example['history'] = ''
        return example
        
    if example['parent_asin'] not in item2meta:
        example['history'] = ''
        
    history = example['history'].split(' ')
    filtered_history = [_ for _ in history if _ in item2meta]
    example['history'] = ' '.join(filtered_history)
    return example


def truncate_history(example, max_his_len):
    example['history'] = ' '.join(example['history'].split(' ')[-max_his_len:])
    return example


def remap_id(datasets):
    user2id = {'[PAD]': 0}
    id2user = ['[PAD]']
    item2id = {'[PAD]': 0}
    id2item = ['[PAD]']

    for split in ['train', 'valid', 'test']:
        dataset = datasets[split]
        for user_id, item_id, history in zip(dataset['user_id'], dataset['parent_asin'], dataset['history']):
            if user_id not in user2id:
                user2id[user_id] = len(id2user)
                id2user.append(user_id)
            if item_id not in item2id:
                item2id[item_id] = len(id2item)
                id2item.append(item_id)
            items_in_history = history.split(' ')
            for item in items_in_history:
                if item not in item2id:
                    item2id[item] = len(id2item)
                    id2item.append(item)

    data_maps = {'user2id': user2id, 'id2user': id2user, 'item2id': item2id, 'id2item': id2item}
    return data_maps


def list_to_str(l):
    if isinstance(l, list):
        return list_to_str(', '.join(l))
    else:
        return l


def clean_text(raw_text):
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text=re.sub(r'[^\x00-\x7F]', ' ', text)
    return text


def feature_process(feature):
    sentence = ""
    if isinstance(feature, float):
        sentence += str(feature)
        sentence += '.'
    elif isinstance(feature, list) and len(feature) > 0:
        for v in feature:
            sentence += clean_text(v)
            sentence += ', '
        sentence = sentence[:-2]
        sentence += '.'
    else:
        sentence = clean_text(feature)
    return sentence + ' '

# def get_normalized_price(price_str:float) -> float:
#     """Helper function to convert price string to a normalized float."""
#     try:
#         price_val = float(price_str)
#         return np.log1p(price_val) 
#     except (ValueError, TypeError):
#         return 0.0


# def get_rating_number(rating_str):
#     try:
#         return np.log(rating_str)
#     except (ValueError, TypeError):
#         return 0


def clean_metadata(example):
    meta_text = ''
    features_needed = ['title', 'features', 'categories', 'description']
    for feature in features_needed:
        meta_text += feature_process(example[feature])
    example['cleaned_metadata'] = meta_text
    # normalized_price = get_normalized_price(example.get('price'))
    # rating_count = get_rating_number(example.get('rating_number'))
    # avg_rating = example.get('average_rating',0)
    # This field will be concatenated with other numerical features in the Item Tower MLP
    # example['numerical_features'] = [normalized_price, rating_count,avg_rating]
    return example

def process_meta(meta_dataset,args):
    meta_dataset = meta_dataset.map(
        clean_metadata,
        num_proc=args.n_workers
    )

    item2meta ={}
    
    for parent_asin, cleaned_metadata in zip(
        meta_dataset['parent_asin'], 
        meta_dataset['cleaned_metadata']
    ):
        # The value is a list containing the two processed features
        # item2meta[parent_asin] = [cleaned_metadata, numerical_features]
        item2meta[parent_asin] = cleaned_metadata

    return item2meta

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def generate_ft(metadata,datasets,args):
    item2meta = process_meta(metadata,args)

    truncated_datasets = {}
    output_dir = os.path.join(args.output_dir, args.domain)
    check_path(output_dir)
    for split in ['train', 'valid', 'test']:
        # Remove lines w/ empty history
        filtered_dataset = datasets[split].map(
            lambda t: filter_items_wo_metadata(t, item2meta),
            num_proc=args.n_workers
        )


        filtered_dataset = filtered_dataset.filter(lambda t: len(t['history']) > 0)
        # Truncate history
        truncated_dataset = filtered_dataset.map(
            lambda t: truncate_history(t, args.max_his_len),
            num_proc=args.n_workers
        )
        
        truncated_datasets[split] = truncated_dataset
        output_path = os.path.join(output_dir, f'{args.domain}.{split}.inter')
        with open(output_path, 'w') as f:
            f.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
            for user_id, history, parent_asin in zip(
                truncated_dataset['user_id'],
                truncated_dataset['history'],
                truncated_dataset['parent_asin']
            ):
                f.write(f"{user_id}\t{history}\t{parent_asin}\n")
                
    '''
    Remap IDs
    '''
    data_maps = remap_id(truncated_datasets)
    id2meta = {0: '[PAD]'}
    for item in item2meta:
        if item not in data_maps['item2id']:
            continue
        item_id = data_maps['item2id'][item]
        id2meta[item_id] = item2meta[item]
    data_maps['id2meta'] = id2meta
    output_path = os.path.join(output_dir, f'{args.domain}.data_maps')
    with open(output_path, 'w') as f:
        json.dump(data_maps, f)

    '''
    Generate item features
    '''
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.plm)
    model = AutoModel.from_pretrained(args.plm).to(device)
    sorted_text = []    # 1-base, sorted_text[0] -> item_id=1
    for i in range(1, len(data_maps['item2id'])):
        sorted_text.append(data_maps['id2meta'][i])
    
    all_embeddings = []
    for pr in tqdm(range(0, len(sorted_text), args.batch_size)):
        batch = sorted_text[pr:pr + args.batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_embeddings.tofile(os.path.join(output_dir, f'{args.domain}.{args.plm.split("/")[-1]}.feature'))

    del model
    del tokenizer
    if device.type == 'cuda':
        torch.cuda.empty_cache() 
    gc.collect() 
    
    '''
    Statistics
    '''
    print(f"#Users: {len(data_maps['user2id']) - 1}")
    print(f"#Items: {len(data_maps['item2id']) - 1}")
    n_interactions = {}
    for split in ['train', 'valid', 'test']:
        n_interactions[split] = len(truncated_datasets[split])
        for history in truncated_datasets[split]['history']:
            if len(history.split(' ')) == 1:
                n_interactions[split] += 1
    print(f"#Interaction in total: {sum(n_interactions.values())}")
    print(n_interactions)
    avg_his_length = 0
    for split in ['train', 'valid', 'test']:
        avg_his_length += sum([len(_.split(' ')) for _ in truncated_datasets[split]['history']])
    avg_his_length /= sum([len(truncated_datasets[split]) for split in ['train', 'valid', 'test']])
    print(f"Average history length: {avg_his_length}")
    print(f"Average character length of metadata: {np.mean([len(_) for _ in sorted_text])}")