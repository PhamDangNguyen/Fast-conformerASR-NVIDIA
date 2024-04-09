python /home/pdnguyen/fast_confomer_finetun/Fast-conformerASR-NVIDIA/process_asr_text_tokenizer.py \
        --data_file="/home/pdnguyen/fast_confomer_finetun/Fast-conformerASR-NVIDIA/metadata_tokenizer/teranscrip_all.json" \
        --data_root="/home/pdnguyen/fast_confomer_finetun/Fast-conformerASR-NVIDIA/dict_N" \
        --vocab_size=10000 \
        --tokenizer="spe" \
        --no_lower_case \
        --spe_type="bpe" \
        --spe_character_coverage=1.0 \
        --log
