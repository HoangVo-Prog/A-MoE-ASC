import os
from src.core.utils.helper import (
    get_config,  
    set_seed,
)
from src.core.utils.const import FUSION_METHOD_CHOICES
from src.core.run import run_benchmark_fusion, train_multi_seed, train_kfold


def main():
    config = get_config()    
    set_seed()
    
    # Benchmarking fusion methods
    if config.benchmark_fusion:
        if not config.benchmark_methods:
           config.benchmark_methods = FUSION_METHOD_CHOICES 
           
        out_path = os.path.join(config.base.output_dir, "fusion_benchmark_results.json")
        
        run_benchmark_fusion(config=config)
        
        print(f"Fusion benchmark results saved to {out_path}")
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