python train_kenlm.py nemo_model_file=/home/pdnguyen/fast_confomer_finetun/train_kenLM/scripts/asr_language_modeling/models/219.nemo \
                          train_paths=[/home/pdnguyen/fast_confomer_finetun/train_kenLM/scripts/asr_language_modeling/merge_text/document.txt] \
                          kenlm_bin_path=/home/pdnguyen/Format_repo/asr-training/Fastconformer/Train_kenLM/NeMo/decoders/kenlm/build/bin \
                          kenlm_model_file=/home/pdnguyen/Format_repo/asr-training/Fastconformer/Train_kenLM/NeMo/scripts/asr_language_modeling/ngram_lm/output_model/test.binary \
                          ngram_length=6 \
                          preserve_arpa=true