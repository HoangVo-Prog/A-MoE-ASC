export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1

python src/moe/train_cli.py \
    --train_path dataset/asc/rest14/train.json \
    --test_path dataset/asc/rest14/test.json \
    --val_path dataset/asc/rest14/dev.json \
    --model_name bert-base-uncased \
    --fusion_method concat \
    --use_moe \
    --freeze_base \
    --route_mask_pad_tokens \
    --aux_loss_weight 0.01
