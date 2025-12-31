import os
from src.core.utils.helper import (
    get_config,  
    get_tokenizer,
    get_dataset,
    get_dataloader,
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
    
    os.makedirs(config.base.output_dir, exist_ok=True)
    
    tokenizer = get_tokenizer(config)
    full_train_set, test_set = get_dataset(config, tokenizer)
    seeds = [config.kfold.seed + i for i in range(config.base.num_seeds)]
    
    label2id = full_train_set.label2id
    id2label = {v: k for k, v in label2id.items()}
    
    full_train_dataloader, _, test_loader = get_dataloader(
        config,
        train_set=full_train_set,
        test_set=test_set,
    )
    
    # Train full only with multi-seed
    if config.base.train_full_only:
        train_multi_seed(
            config,
            methods=config.base.fusion_method,
            full_train_dataloader=full_train_dataloader,
            test_loader=test_loader,
            label2id=label2id,
            id2label=id2label,
            seeds=seeds,
            print_cf_matrix=False,
            do_ensemble_logits=config.base.do_ensemble_logits,
            verbose_ensemble_report=False,
        )
        print("Training with multiple seeds completed.")
        return
        
    
    # Train with k-fold cross validation 
    train_kfold(config, methods=config.base.fusion_method)
    print("Training with k-fold cross validation")


if __name__ == "__main__":
    main()