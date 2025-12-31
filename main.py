import os
from src.core.utils.helper import (
    get_config,  
    set_seed,
)
from src.core.utils.const import FUSION_METHOD_CHOICES
from src.core.run import run_benchmark_fusion, train_multi_seed, train_kfold


def main():
    config = get_config() 
    print("Configuration:", config)   
    set_seed()
    
    # Benchmarking fusion methods
    if config.base.benchmark_fusion:
        if not config.base.benchmark_methods:
           config.base.benchmark_methods = FUSION_METHOD_CHOICES 
                   
        run_benchmark_fusion(config=config)
        print("Benchmarking fusion methods completed.")
        return
    
    # Train full only with multi-seed
    if config.base.train_full_only:
        train_multi_seed(config)
        print("Training with multiple seeds completed.")
        return
        
    
    # Train with k-fold cross validation 
    train_kfold(config, methods=config.base.fusion_methods)
    print("Training with k-fold cross validation")


if __name__ == "__main__":
    main()