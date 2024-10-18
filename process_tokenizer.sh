python  process_asr_text_tokenizer.py \
        --manifest="/home/pdnguyen/Format_repo/asr-training/Fastconformer/metadata_train/val.json" \
        --data_root="/home/pdnguyen/Format_repo/asr-training/Fastconformer/dict_N" \
        --vocab_size=1024 \
        --tokenizer="spe" \
        --no_lower_case \
        --spe_type="bpe" \
        --spe_character_coverage=1.0 \
        --log