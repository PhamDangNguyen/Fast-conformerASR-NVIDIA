# N-gram Language Modeling
In this approach, an N-gram LM is trained on text data, NeMo supports both character-based and BPE-based models for N-gram LMs. An N-gram LM can be used with beam search decoders on top of the ASR models to produce more accurate candidates. The beam search decoders in NeMo support language models trained with KenLM (refer [Detail about train kenLM](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/asr_language_modeling_and_customization.html)
). 

In this repo you can train kenLM following:

## Training KenLM
1. In the first step, install the necessary libraries and dependencies
```
cd Fastconformer/Train_kenLM
pip install -r requirements.txt
sh install_beam.sh
```
2. In the second step, train the kenLM model
```
cd /Fastconformer/Train_kenLM/NeMo/scripts/asr_language_modeling/ngram_lm
```
Before training, you need to downgrade the numpy version to 1.x to train the KenLM model.
```
pip uninstall numpy
pip install numpy==1.26.4
```
After, in ``train_kenlm.sh`` file you have to fill:
```
python train_kenlm.py \
    nemo_model_file=...  # Path to the .nemo ASR model \
    train_paths=...      # Path to the text corpus [.txt file] \
    kenlm_bin_path=/Fastconformer/Train_kenLM/NeMo/decoders/kenlm/build/bin \
    kenlm_model_file=... # Output path for the .binary model \
    ngram_length=6       # Default is 6; you can adjust to 3, 4, etc. for speed-accuracy trade-offs \
    preserve_arpa=true   # Set to true to keep the ARPA file
```
Start training
```
sh train_kenlm.sh
```
## Perform inference using the ASR .nemo model combined with the LM model
Test infer using code
```
cd Fastconformer/Train_kenLM/infer_with_kenLM
sh infer_LM.sh
```
With fields you can change it

    --nemo_model: ....... # the .nemo ASR file
    --kenlm_model: ...... # the .binary kenLM file