import yaml
import os
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(__file__))

CONFIG_PATH = os.path.join(ROOT, "configs", "twotower_config.yaml")

# load YAML
with open(CONFIG_PATH, "r") as f:
    _raw_config: Dict[str, Any] = yaml.safe_load(f)

PROCESSED_DIR: str = _raw_config["preprocess"]["PROCESSED_DIR"]

# train
NUM_EPOCHS: int = _raw_config["train"]["num_epochs"]
BATCH_SIZE: int = _raw_config["train"]["batch_size"]
LR: float = _raw_config["train"]["lr"]

# model
EMBEDDING_DIM: int = _raw_config["model"]["embedding_dim"]

# eval
TOPK_LIST: List[int] = _raw_config["eval"]["topk_list"]