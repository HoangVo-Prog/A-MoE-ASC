# src/main.py
import os
from src.core.config import Config
from src.core.utils.helper import set_seed
from src.core.run import run_single_train_eval

def main():
    cfg = Config.from_cli().finalize().validate()
    
    print("Configuration:")
    print(cfg)
    print()

    set_seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    run_single_train_eval(cfg, method=cfg.fusion_method)

if __name__ == "__main__":
    main()
