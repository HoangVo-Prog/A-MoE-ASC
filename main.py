# src/main.py
import os
from src.core.config import Config
from src.core.utils.helper import set_seed, get_tokenizer, get_dataset, get_dataloader
from src.core.run import run_benchmark_fusion, train_multi_seed, train_kfold

def main():
    cfg = Config.from_cli().finalize().validate()
    
    print("Configuration:")
    print(cfg)
    print()

    set_seed(cfg.seed)

    if cfg.is_benchmark:
        run_benchmark_fusion(config=cfg)
        return

    os.makedirs(cfg.output_dir, exist_ok=True)

    tokenizer = get_tokenizer(cfg)
    full_train_set, test_set = get_dataset(cfg, tokenizer)

    label2id = full_train_set.label2id
    id2label = {v: k for k, v in label2id.items()}
    cfg.label2id = label2id
    cfg.id2label = id2label

    full_train_loader, _, test_loader = get_dataloader(cfg, train_set=full_train_set, test_set=test_set)

    if cfg.train_full_only:
        train_multi_seed(
            config=cfg,
            method=cfg.fusion_method,
            full_train_dataloader=full_train_loader,
            test_loader=test_loader,
            label2id=label2id,
            id2label=id2label,
            seeds=cfg.seed_list,
            print_cf_matrix=False,
            do_ensemble_logits=cfg.do_ensemble_logits,
            verbose_ensemble_report=False,
        )
        return

    train_kfold(cfg, method=cfg.fusion_method)

if __name__ == "__main__":
    main()
