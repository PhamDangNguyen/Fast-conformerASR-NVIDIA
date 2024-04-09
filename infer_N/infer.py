import nemo.collections.asr as nemo_asr

# Load the ASR model checkpoint
model_ckpt = "a.ckpt"
model = nemo_asr.models.EncDecCTCModelBPE.load_from_checkpoint(model_ckpt)

print(model.transcribe(['0_1ch.wav']))
