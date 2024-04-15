import nemo.collections.asr as nemo_asr
import torch
# Load the ASR model checkpoint
model_ckpt = "0.0227_wer.ckpt"
model = nemo_asr.models.EncDecCTCModelBPE.load_from_checkpoint(model_ckpt)
if torch.cuda.is_available():
    # Chuyển model và dữ liệu sang GPU
    device = torch.device("cuda")
    model.to(device)
    print("Model đang sử dụng GPU")
else:
    print("CUDA không có sẵn, model sẽ sử dụng CPU")
#infer batch
predicted_text = model.transcribe(
    paths2audio_files=['0_1ch.wav', '2_1ch.wav'],
    batch_size=2  # batch size to run the inference with
)
print(predicted_text)