import torch
import os
import numpy as np
import random
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from src.core.data.datasets import AspectSentimentDataset, AspectSentimentDatasetKFold
from src.core.utils.general import cfg_to_flat_dict, build_kwargs_from_signature
from src.core.utils.const import DEVICE
from importlib import import_module


def get_model(cfg):
    """
    Chuẩn chỉnh:
    - cfg.mode chọn class trong src.models
    - Không fallback (không cfg.base/cfg.moe/cfg.kfold)
    - kwargs chỉ lấy theo signature __init__
    - Báo lỗi rõ nếu thiếu required args
    """
    mode = str(getattr(cfg, "mode", "")).strip()
    if not mode:
        raise ValueError("cfg.mode is empty")

    models_mod = import_module("src.models")

    if not hasattr(models_mod, mode):
        public = [n for n in dir(models_mod) if n and n[0].isupper()]
        raise ValueError(f"Unknown cfg.mode='{mode}'. Available: {public}")

    ModelCls = getattr(models_mod, mode)

    cfg_dict = cfg_to_flat_dict(cfg)
    kwargs = build_kwargs_from_signature(cfg_dict, ModelCls)

    model = ModelCls(**kwargs).to(DEVICE)
    return model

def get_kfold_dataset(cfg, tokenizer):
    return AspectSentimentDatasetKFold(
            json_path=cfg.train_path,
            tokenizer=tokenizer,
            max_len_sent=cfg.max_len_sent,
            max_len_term=cfg.max_len_term,
            k_folds=cfg.k_folds,
            seed=cfg.seed,
            shuffle=cfg.shuffle,
            debug_aspect_span=getattr(cfg, "debug_aspect_span", False),
            
        )

def get_dataset(cfg, tokenizer):
    train_set = AspectSentimentDataset(
        json_path=cfg.train_path,
        tokenizer=tokenizer,
        max_len_sent=cfg.max_len_sent,
        max_len_term=cfg.max_len_term,
        debug_aspect_span=getattr(cfg, "debug_aspect_span", False),
    )
    
    test_set = AspectSentimentDataset(
        json_path=cfg.test_path,
        tokenizer=tokenizer,
        max_len_sent=cfg.max_len_sent,
        max_len_term=cfg.max_len_term,
        debug_aspect_span=getattr(cfg, "debug_aspect_span", False),
    )  

    return train_set, test_set 

def get_dataloader(cfg, train_set=None, val_set=None, test_set=None):
    
    train_loader, val_loader, test_loader = None, None, None
    
    if train_set:
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    
    if val_set:
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    
    if test_set:
        test_loader = DataLoader(
            test_set,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader

def get_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)    
    return tokenizer

def set_seed(seed: int = 13):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    
