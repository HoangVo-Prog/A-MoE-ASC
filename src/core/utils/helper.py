import torch
import os
import numpy as np
from dataclasses import fields, is_dataclass
import random
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from src.core.config import Config
from src.core.cli import parse_args
from src.core.data.datasets import AspectSentimentDataset, AspectSentimentDatasetKFold
from src.core.utils.general import to_dict, infer_store_true_dests, filter_config_kwargs





def get_arg_parser_parameters(
    args,
    config_cls,
    *,
    arg_parser=None,
    drop_false_store_true=True,
) -> dict:
    cfg_default = config_cls()

    out = {}
    for name, sub_cfg in cfg_default.__dict__.items():
        if is_dataclass(sub_cfg) and not isinstance(sub_cfg, type):
            sub_cls = type(sub_cfg)
            sub_kwargs = filter_config_kwargs(
                args,
                sub_cls,
                arg_parser=arg_parser,
                drop_false_store_true=drop_false_store_true,
            )
            out[name] = sub_cls(**sub_kwargs)
        else:
            pass

    return out



def get_config(args=parse_args()):
    return Config(**get_arg_parser_parameters(args, Config))


def get_model_parameter(
    cfg,
    model_cls_or_callable,
) -> dict:
    """
    Trả kwargs để khởi tạo model từ cfg theo signature của model.

    Usage:
        kwargs = get_model_parameter(cfg, ModelCls, fallback_sources=[cfg.base, cfg.moe, cfg.kfold])
        model = ModelCls(**kwargs)

    Ghi chú:
    - cfg có thể là dataclass (Config) hoặc mapping/namespace tùy _to_dict hỗ trợ.
    - fallback_sources dùng để lấy param bị thiếu từ cfg.base/cfg.moe/cfg.kfold.
    """
    if fallback_sources is None:
        # default theo style bạn đang dùng
        fallback_sources = [getattr(cfg, "base", None), getattr(cfg, "moe", None), getattr(cfg, "kfold", None)]
        fallback_sources = [x for x in fallback_sources if x is not None]

    return filter_config_kwargs(
        cfg,
        model_cls_or_callable,
        fallback_sources=fallback_sources,
    )


def get_model(cfg):
    mode = cfg.base.mode
    ModelCls = getattr(__import__("src.models", fromlist=[mode]), mode)

    kwargs = get_model_parameter(
        cfg,
        ModelCls,
    )
    return ModelCls(**kwargs)



def get_kfold_dataset(cfg, tokenizer):
    return AspectSentimentDatasetKFold(
            config=cfg,
            json_path=cfg.base.train_path,
            tokenizer=tokenizer,
            max_len_sent=cfg.base.max_len_sent,
            max_len_term=cfg.base.max_len_term,
        )

def get_dataset(cfg, tokenizer):
    train_set = AspectSentimentDataset(
        json_path=cfg.base.train_path,
        tokenizer=tokenizer,
        max_len_sent=cfg.base.max_len_sent,
        max_len_term=cfg.base.max_len_term,
    )
    
    test_set = AspectSentimentDataset(
        json_path=cfg.base.test_path,
        tokenizer=tokenizer,
        max_len_sent=cfg.base.max_len_sent,
        max_len_term=cfg.base.max_len_term,
    )  

    return train_set, test_set 

def get_dataloader(cfg, train_set=None, val_set=None, test_set=None):
    
    train_loader, val_loader, test_loader = None, None, None
    
    if train_set:
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.base.train_batch_size,
            shuffle=True,
            num_workers=cfg.base.num_workers,
            pin_memory=True,
        )
    
    if val_set:
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.base.eval_batch_size,
            shuffle=False,
            num_workers=cfg.base.num_workers,
            pin_memory=True,
        )
    
    if test_set:
        test_loader = DataLoader(
            test_set,
            batch_size=cfg.base.test_batch_size,
            shuffle=False,
            num_workers=cfg.base.num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader

def get_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.base.model_name)    
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
    