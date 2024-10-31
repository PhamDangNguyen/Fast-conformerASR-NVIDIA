python train_kenlm.py nemo_model_file=/home/team_voice/STT_pdnguyen/Backup_version_train/fast_conformer/fubon_2024_2_10_6GPU/nemo_experiments/FastConformer-CTC-BPE/checkpoints/FastConformer-CTC-BPE.nemo \
                          train_paths=[/home/team_voice/STT_pdnguyen/Backup_version_train/fast_conformer/fubon_2024_2_10_6GPU/dict_N/text_corpus/document.txt] \
                          kenlm_bin_path=/home/team_voice/STT_pdnguyen/asr-training/Fastconformer/Train_kenLM/NeMo/decoders/kenlm/build/bin \
                          kenlm_model_file=/home/team_voice/STT_pdnguyen/asr-training/Fastconformer/Train_kenLM/NeMo/scripts/asr_language_modeling/ngram_lm/output_model/test_1.binary \
                          ngram_length=6 \
                          preserve_arpa=true