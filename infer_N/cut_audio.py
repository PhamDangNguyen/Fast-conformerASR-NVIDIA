import os
path_audio = "/home/pdnguyen/fast_confomer_finetun/finetune-fast-conformer/infer_N/chunk_5_normalized.wav"
cmd = f"ffmpeg -i {path_audio} -ss 1.723 -to 2.325 /home/pdnguyen/fast_confomer_finetun/finetune-fast-conformer/infer_N/cut_audio_to_check/CMND.wav"
os.system(cmd)