# Feature Preprocessing

## Dataset
Source: Amazon-reviews-2023

Category: Electronics

Pre-split (5core-timestamp) dataset is used. Additionally, due to resources constraint, we downsampled the dataset (`shrink_ratio`:0.05, current_dataset_size = 0.05*original_dataset_size) 

## User Tower
**Features Included**

due to time constraints, only `user_id` is used input feautures included in user tower for now.

**Improvement plan**
1. sequential history 

* Item Sequence: The sequence of item IDs previously interacted with is extracted for each user.

* Sequence Encoder: This sequence is fed into a Sequential Recommender Model architecture, often a Transformer-based model (like SASRec or BERT4Rec) or an RNN

2. Review Text

* Each past review is encoded by a Text Encoder (like BLaIR, though this adds complexity).The resulting review embeddings are then aggregated (e.g., by averaging or using an attention mechanism) to create a Textual User Preference Vector.


## Item Tower
**Features Included**

`parent_asin`: product id (identifier)

text-based features inlcuding `title`, `features`, `description`, `datails` 

we embedded test-based features by using pre-trained model BLaiR which is specifically trained on Amazon-reviews-2023.




