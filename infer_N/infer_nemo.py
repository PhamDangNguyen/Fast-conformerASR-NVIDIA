from dataclasses import dataclass, field
from typing import List, Optional
import soundfile as sf
import logging
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
import torch
from omegaconf import DictConfig, MISSING
import torch
import tempfile
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.submodules import ctc_beam_decoding
from nemo.collections.asr.parts.utils.transcribe_utils import TextProcessingConfig
from nemo.utils import logging
from nemo.collections.asr.models import EncDecCTCModelBPE
import json
from tqdm.auto import tqdm
import os






model_nemo = "/home/pdnguyen/fast_confomer_finetun/finetune-fast-conformer/models_convert/asr_model.nemo"
model = EncDecCTCModelBPE.restore_from(model_nemo, map_location=torch.device("cuda"))


predicted_text = model.transcribe(
    paths2audio_files=[
        '/home/pdnguyen/fast_confomer_finetun/finetune-fast-conformer/infer_N/fubong/20240222164216_giọng nữ miền Bắc 1/chunk_13_normalized.wav'
        ],
    batch_size=2  # batch size to run the inference with
)
print(predicted_text)
