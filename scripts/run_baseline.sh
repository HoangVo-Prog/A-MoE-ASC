!python src/baseline/train.py \
    --model_name roberta-base \
    --train_path /kaggle/working/A-MoE-ASC/dataset/asc/rest14/train.json \
    --test_path /kaggle/working/A-MoE-ASC/dataset/asc/rest14/test.json \
    --val_path /kaggle/working/A-MoE-ASC/dataset/asc/rest14/dev.json \
    --fusion_method concat