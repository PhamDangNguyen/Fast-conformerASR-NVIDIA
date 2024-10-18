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

predicted_text = model.transcribe(
    paths2audio_files=[
        'infer_N/short/nam bình định/nam bình định_60.2_73.92.wav',
        ],
    batch_size=2  # batch size to run the inference with
)
print(predicted_text, len(predicted_text[0].split(" ")))

#infer batch
predicted_text = model.transcribe(
    paths2audio_files=[
        'infer_N/1709825057_merge.wav',
        "infer_N/0_1ch.wav",
        "infer_N/2_1ch.wav",
        "infer_N/cong_nghe_1_000011.wav",
        "infer_N/short/226_002_0.18_14.04.wav", 
        "infer_N/short/226_002_14.44_28.4.wav",
        "infer_N/short/4_000002.wav"
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
