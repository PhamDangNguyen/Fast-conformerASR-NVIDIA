import nemo.collections.asr as nemo_asr
import torch
# Load the ASR model checkpoint
model_ckpt = "/mnt/driver/Back_up/models/6_12_2024_all_2023_2024_data/nemo_experiments/FastConformer-CTC-BPE/checkpoints/FastConformer-CTC-BPE--val_wer=0.0973-epoch=188.ckpt"
model = nemo_asr.models.EncDecCTCModelBPE.load_from_checkpoint(model_ckpt)
if torch.cuda.is_available():
    # Chuyển model và dữ liệu sang GPU
    device = torch.device("cuda")
    model.to(device)
    print("Model đang sử dụng GPU")
    
#infer batch
predicted_text = model.transcribe(
    paths2audio_files=[
        'wavs_test/output_segment_0.wav',
        "wavs_test/output_segment_1.wav",
        ],
    batch_size=2  # batch size to run the inference with
)
print(predicted_text)
from pathlib import Path
files = Path("infer_N/short").rglob("*.wav")

for file in files:
    print(file)
    predicted_text = model.transcribe(
        paths2audio_files=[str(file)],
        batch_size=1  # batch size to run the inference with
    )
    print(predicted_text)
    print()
