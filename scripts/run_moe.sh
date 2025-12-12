python src/moe/train_cli \
    --train_path /kaggle/working/A-MoE-ASC/dataset/asc/rest14/train.json \
    --test_path /kaggle/working/A-MoE-ASC/dataset/asc/rest14/test.json \
    --val_path /kaggle/working/A-MoE-ASC/dataset/asc/rest14/dev.json \
    --model_name bert-base-uncased \
    --use_moe \
    --freeze_base \
    --route_mask_pad_tokens \
    --aux_loss_weight 0.01
