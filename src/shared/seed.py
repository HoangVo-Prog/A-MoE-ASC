import random
import torch
import numpy as np
import os
from torch.utils.data import DataLoader


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

def set_determinism(seed: int) -> None:
    """Best-effort determinism for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    
    
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    
def make_train_loader_with_seed(train_dataset_full, batch_size: int, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        train_dataset_full,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )

