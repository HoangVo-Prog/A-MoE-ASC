import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix

from src.core.utils.helper import (
    get_tokenizer,
    get_dataset,
    get_dataloader,
    get_kfold_dataset,
    get_model,
    set_seed,
)
from src.core.utils.general import (
    cleanup_cuda,
    collect_test_logits,
    mean_std,
    logits_to_metrics,
    aggregate_confusions,
    parse_str_list
)
from .engine import run_training_loop, eval_model
from .train import train_multi_seed


def run_benchmark_fusion(config):
    os.makedirs(config.output_dir, exist_ok=True)
    
    tokenizer = get_tokenizer(config)
    full_train_set, test_set = get_dataset(config, tokenizer)
    seeds = [config.seed + i for i in range(config.num_seeds)]
    
    label2id = full_train_set.label2id
    id2label = {v: k for k, v in label2id.items()}
    samples = full_train_set.samples
    y = [label2id[s["sentiment"]] for s in samples]
    num_classes = len(label2id)
    
    kfold_train_set = get_kfold_dataset(config, tokenizer)
    full_train_dataloader, _, test_loader = get_dataloader(config, train_set=full_train_set, test_set=test_set)
    
    methods = parse_str_list(config.benchmark_methods)
    
    all_results = {
        "methods": methods,
        "seeds": seeds,
        "folds": config.k_folds,
        "runs": {},
        "summary": {},
        "full_confusion": {},
        "ensemble": {},
    }
    
    per_method_seed_records = {m: [] for m in methods}
    
    if config.mode == "MoFModel":
        methods = "MoF"
    
    for method in methods:
        per_method_seed_records[method] = []
        seed_oof_logits_list = []
        seed_test_logits_list = [] 
        
        for seed in seeds:
            oof_logits = None
            oof_filled = None  
            fold_test_logits = []
            set_seed(seed)
            
            fold_val_acc = []
            fold_val_f1 = []
            fold_test_acc = []
            fold_test_f1 = []
            fold_val_cms = []
            fold_test_cms = []      
            
            for fold in range(config.k_folds):                
                train_ds, val_ds = kfold_train_set.get_fold(fold)
                train_loader, val_loader, _ = get_dataloader(cfg=config, train_set=train_ds, val_set=val_ds) 
                
                model = get_model(config)
                
                out = run_training_loop(
                    cfg=config,
                    model=model,
                    method=method,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    id2label=id2label,
                    tag=f"[CV {method} seed={seed} fold={fold}] ",
                )
                
                best_sd = out.get("best_state_dict", None)
                if best_sd is not None:
                    model.load_state_dict(best_sd)

                # Collect logits for OOF (val) + test to enable ensemble
                val_logits, val_labels = collect_test_logits(
                    model=model,
                    test_loader=val_loader,
                    fusion_method=method
                )
                test_logits, test_labels = collect_test_logits(
                    model=model,
                    test_loader=test_loader,
                    fusion_method=method,
                )
                va_idx = np.asarray(val_ds.base_indices, dtype=np.int64)

                if oof_logits is None:
                    num_train = len(full_train_set.samples)
                    num_classes = int(val_logits.shape[1])
                    oof_logits = np.zeros((num_train, num_classes), dtype=np.float32)
                    oof_filled = np.zeros((num_train,), dtype=bool)

                oof_logits[va_idx] = val_logits.astype(np.float32)
                oof_filled[va_idx] = True
                fold_test_logits.append(test_logits.astype(np.float32))
                
                val_m = eval_model(
                        model=model,
                        dataloader=val_loader,
                        id2label=id2label,
                        verbose_report=False,
                        print_cf_matrix=False,
                        fusion_method=method,
                        f1_average="macro",
                        return_confusion=True,
                    )
                    
                test_m = eval_model(
                    model=model,
                    dataloader=test_loader,
                    id2label=id2label,
                    verbose_report=False,
                    print_cf_matrix=False,
                    fusion_method=method,
                    f1_average="macro",
                    return_confusion=True,
                )
                    
                print(
                    f"[FINAL] Test loss {test_m['loss']:.4f} "
                    f"F1 {test_m['f1']:.4f} "
                    f"acc {test_m['acc']:.4f} "
                )

                fold_val_acc.append(float(val_m["acc"]))
                fold_test_acc.append(float(test_m["acc"]))
                fold_val_f1.append(float(val_m["f1"]))
                fold_test_f1.append(float(test_m["f1"]))
                fold_val_cms.append(np.asarray(val_m["confusion"], dtype=np.float64))
                fold_test_cms.append(np.asarray(test_m["confusion"], dtype=np.float64))


                del model
                cleanup_cuda()

            cv_val_mean, cv_val_std = mean_std(fold_val_f1)
            cv_test_mean, cv_test_std = mean_std(fold_test_f1)

            # Per-seed ensembles
            if not bool(oof_filled.all()):
                missing = int((~oof_filled).sum())
                raise RuntimeError(f"OOF logits not fully filled (missing={missing})")

            y_true = np.asarray(y, dtype=int)
            oof_metrics = logits_to_metrics(oof_logits, y_true)
            oof_preds = oof_logits.argmax(axis=-1)
            oof_cm = confusion_matrix(y_true, oof_preds, labels=list(range(num_classes)))

            seed_test_ens_logits = np.mean(np.stack(fold_test_logits, axis=0), axis=0)
            seed_test_metrics = logits_to_metrics(seed_test_ens_logits, test_labels)
            seed_test_preds = seed_test_ens_logits.argmax(axis=-1)
            seed_test_cm = confusion_matrix(test_labels, seed_test_preds, labels=list(range(num_classes)))

            # keep logits for seed-level ensemble across seeds
            seed_oof_logits_list.append(oof_logits.astype(np.float32))
            seed_test_logits_list.append(seed_test_ens_logits.astype(np.float32))

            record = {
                "fusion_method": method,
                "seed": int(seed),
                "cv_val_acc_folds": fold_val_acc,
                "cv_val_f1_folds": fold_val_f1,
                "cv_test_acc_folds": fold_test_acc,
                "cv_test_f1_folds": fold_test_f1,
                "cv_val_f1_mean": float(cv_val_mean),
                "cv_val_f1_std": float(cv_val_std),
                "cv_test_f1_mean": float(cv_test_mean),
                "cv_test_f1_std": float(cv_test_std),
                "cv_val_confusion": aggregate_confusions(fold_val_cms),
                                    "cv_val_oof_ens_acc": float(oof_metrics["acc"]),
                "cv_val_oof_ens_f1": float(oof_metrics["f1"]),
                "cv_val_oof_ens_confusion": aggregate_confusions([oof_cm]),
                "cv_test_ens_acc": float(seed_test_metrics["acc"]),
                "cv_test_ens_f1": float(seed_test_metrics["f1"]),
                "cv_test_ens_confusion": aggregate_confusions([seed_test_cm]),
            }
            per_method_seed_records[method].append(record)

        # Benchmark-level ensemble across seeds (using OOF logits for CV-val, and seed-ensembled logits for test)
        if len(seed_oof_logits_list) >= 2:
            ens_oof = np.mean(np.stack(seed_oof_logits_list, axis=0), axis=0)
            y_true = np.asarray(y, dtype=int)
            m_oof = logits_to_metrics(ens_oof, y_true)
            p_oof = ens_oof.argmax(axis=-1)
            cm_oof = confusion_matrix(y_true, p_oof, labels=list(range(num_classes)))

            ens_test = np.mean(np.stack(seed_test_logits_list, axis=0), axis=0)
            # test_labels is same for all seeds, reuse from last fold collection
            m_test = logits_to_metrics(ens_test, test_labels)
            p_test = ens_test.argmax(axis=-1)
            cm_test = confusion_matrix(test_labels, p_test, labels=list(range(num_classes)))

            all_results.setdefault("ensemble", {}).setdefault(method, {})
            all_results["ensemble"][method].update({
                "cv_val_seed_ens_acc": float(m_oof["acc"]),
                "cv_val_seed_ens_f1": float(m_oof["f1"]),
                "cv_val_seed_ens_confusion": aggregate_confusions([cm_oof]),
                "cv_test_seed_ens_acc": float(m_test["acc"]),
                "cv_test_seed_ens_f1": float(m_test["f1"]),
                "cv_test_seed_ens_confusion": aggregate_confusions([cm_test]),
            })

        full_out = train_multi_seed(
            config=config,
            method=method,
            full_train_dataloader=full_train_dataloader,
            test_loader=test_loader,
            label2id=label2id,
            id2label=id2label,
            seeds=seeds,
            print_cf_matrix=False,
            do_ensemble_logits=config.do_ensemble_logits,
            verbose_ensemble_report=False,
        )

        full_by_seed = {int(r["seed"]): r for r in full_out.get("per_seed", [])}
        for rec in per_method_seed_records[method]:
            s = int(rec["seed"])
            if s in full_by_seed:
                rec["full_test_acc"] = float(full_by_seed[s]["acc"])
                rec["full_test_f1"] = float(full_by_seed[s]["f1"])

        all_results["full_confusion"][method] = full_out.get("confusion", {})
        ens = full_out.get("ensemble", None)
        if ens is not None:
            all_results.setdefault("ensemble", {}).setdefault(method, {})
            all_results["ensemble"][method].update({
                "full_ens_test_acc": float(ens["metrics"]["acc"]),
                "full_ens_test_f1": float(ens["metrics"]["f1"]),
                "confusion": ens.get("confusion", None),
            })

    all_results["runs"] = per_method_seed_records

    summary: dict[str, dict] = {}
    for method in methods:
        recs = per_method_seed_records[method]
        cv_val_means = [float(r["cv_val_f1_mean"]) for r in recs]
        cv_test_means = [float(r["cv_test_f1_mean"]) for r in recs]
        full_f1s = [float(r.get("full_test_f1", 0.0)) for r in recs]
        full_accs = [float(r.get("full_test_acc", 0.0)) for r in recs]

        m1, s1 = mean_std(cv_val_means)
        m2, s2 = mean_std(cv_test_means)
        m3, s3 = mean_std(full_f1s)
        m4, s4 = mean_std(full_accs)

        method_sum = {
            "cv_val_f1_mean_over_seeds": float(m1),
            "cv_val_f1_std_over_seeds": float(s1),
            "cv_test_f1_mean_over_seeds": float(m2),
            "cv_test_f1_std_over_seeds": float(s2),
            "full_test_f1_mean_over_seeds": float(m3),
            "full_test_f1_std_over_seeds": float(s3),
            "full_test_acc_mean_over_seeds": float(m4),
            "full_test_acc_std_over_seeds": float(s4),
        }

        if len(recs) > 0 and "cv_test_confusion" in recs[0]:
            cv_val_seed_means = [np.asarray(r["cv_val_confusion"]["cm_mean"], dtype=np.float64) for r in recs]
            cv_test_seed_means = [np.asarray(r["cv_test_confusion"]["cm_mean"], dtype=np.float64) for r in recs]
            method_sum["cv_val_confusion_over_seeds"] = aggregate_confusions(cv_val_seed_means)
            method_sum["cv_test_confusion_over_seeds"] = aggregate_confusions(cv_test_seed_means)

        if method in all_results.get("full_confusion", {}):
            method_sum["full_confusion_over_seeds"] = all_results["full_confusion"][method]

        if method in all_results.get("ensemble", {}):
            method_sum["full_ens_test_acc"] = float(all_results["ensemble"][method]["full_ens_test_acc"])
            method_sum["full_ens_test_f1"] = float(all_results["ensemble"][method]["full_ens_test_f1"])
            if all_results["ensemble"][method].get("confusion", None) is not None:
                method_sum["full_ens_confusion"] = all_results["ensemble"][method]["confusion"]

        summary[method] = method_sum

    if "sent" in summary:
        base = float(summary["sent"]["full_test_f1_mean_over_seeds"])
        for method in methods:
            summary[method]["delta_full_test_f1_vs_sent"] = float(
                float(summary[method]["full_test_f1_mean_over_seeds"]) - base
            )
            if "full_ens_test_f1" in summary[method] and "full_ens_test_f1" in summary["sent"]:
                summary[method]["delta_full_ens_test_f1_vs_sent"] = float(
                    float(summary[method]["full_ens_test_f1"]) - float(summary["sent"]["full_ens_test_f1"])
                )

    all_results["summary"] = summary

    output_path = os.path.join(config.output_dir, config.output_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results
