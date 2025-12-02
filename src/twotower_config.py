import yaml
import os
from typing import Any, Dict, List

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(__file__))

# YAML 配置路径
CONFIG_PATH = os.path.join(ROOT, "configs", "config.yaml")

# 加载 YAML
with open(CONFIG_PATH, "r") as f:
    _raw_config: Dict[str, Any] = yaml.safe_load(f)

# preprocess
DATA_RAW_PATH: str = _raw_config["preprocess"]["DATA_RAW_PATH"]
PROCESSED_DIR: str = _raw_config["preprocess"]["PROCESSED_DIR"]

# train
BATCH_SIZE: int = _raw_config["train"]["batch_size"]
LR: float = _raw_config["train"]["lr"]

# model
EMBEDDING_DIM: int = _raw_config["model"]["embedding_dim"]

# eval
TOPK_LIST: List[int] = _raw_config["eval"]["topk_list"]