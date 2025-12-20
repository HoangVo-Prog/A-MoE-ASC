import torch.nn as nn

def cleanup_cuda(*objs):
    import gc, torch
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = trainable


def maybe_freeze_encoder(model: nn.Module, epoch_idx_0based: int, freeze_epochs: int) -> None:
    if freeze_epochs > 0 and epoch_idx_0based < freeze_epochs:
        set_encoder_trainable(model, False)
    else:
        set_encoder_trainable(model, True)


