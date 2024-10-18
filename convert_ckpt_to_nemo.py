import torch
torch.cuda.empty_cache()
from pytorch_lightning import seed_everything
import copy
seed_everything(42)
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.asr_confidence_utils import (
    ConfidenceConfig,
    ConfidenceConstants,
    ConfidenceMethodConfig,
    ConfidenceMethodConstants,
)

# model_ckpt = "/mnt/driver/Back_up/models/29_5_2024_best_7_13perxent/nemo_experiments/FastConformer-CTC-BPE/checkpoints/FastConformer-CTC-BPE--val_wer=0.0748-epoch=173.ckpt"
# model = nemo_asr.models.EncDecCTCModelBPE.load_from_checkpoint(model_ckpt)
# save_path = f"/home/pdnguyen/fast_confomer_finetun/finetune-fast-conformer/models_convert/173.nemo"
# model.save_to(f"{save_path}")