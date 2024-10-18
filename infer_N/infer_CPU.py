import nemo.collections.asr as nemo_asr
import torch
# Load the ASR model checkpoint
model_ckpt = "/home/pdnguyen/fast_confomer_finetun/finetune-fast-conformer/nemo_experiments/FastConformer-CTC-BPE/checkpoints/FastConformer-CTC-BPE--val_wer=0.1040-epoch=176-last.ckpt"
model = nemo_asr.models.EncDecCTCModelBPE.load_from_checkpoint(model_ckpt,map_location='cpu')
# model = nemo_asr.models.EncDecCTCModelBPE.restore_from(model_ckpt, map_location='cpu')

device = torch.device("cuda")
model.to(device)

# predicted_text = model.transcribe(
#     paths2audio_files=[
#         '/home/pdnguyen/fast_confomer_finetun/finetune-fast-conformer/infer_N/0_1ch.wav'
#         ],
#     batch_size=1  # batch size to run the inference with
# )

# print(predicted_text, len(predicted_text[0].split(" ")))

#infer batch
predicted_text = model.transcribe(
    paths2audio_files=[
        "/home/pdnguyen/fast_confomer_finetun/finetune-fast-conformer/infer_N/fubong/20240222150201_giọng nữ miền nam 2/chunk_2_normalized.wav"
        ],
    batch_size=2  # batch size to run the inference with
    ,return_hypotheses=False
)
print(predicted_text)

