import warnings
warnings.filterwarnings('ignore')
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
import soundfile as sf
import numpy as np
import random
import kenlm
import torch
import time
import tqdm
import json
import os
import re
import argparse

from vad import segmentation

def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)



def read_config(filename):
    with open(filename) as fid:
        return [json.loads(l) for l in fid]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model_type", default="large")
    args = parser.parse_args()


    lm_path = './models/lm/vi_lm_4grams.bin'
    assert os.path.exists(lm_path)

    save_dir = f'output/wav2vec2-{args.model_type}-nguyenvulebinh'

    # load model and tokenizer
    tokenizer = Wav2Vec2CTCTokenizer(
        f'./vocab/vocab_{args.model_type}.json', 
        unk_token='<unk>', 
        pad_token='<pad>', 
        word_delimiter_token=' ' if args.model_type.lower() == 'large' else '|'
    )
    
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, 
        sampling_rate=16000, 
        padding_value=0.0, 
        do_normalize=True, 
        return_attention_mask=False
    )

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    last_checkpoint = get_last_checkpoint(save_dir)
    print(f"last_checkpoint: {last_checkpoint}")

    model = Wav2Vec2ForCTC.from_pretrained(last_checkpoint).to(args.device)

    # beam search
    def get_decoder_ngram_model(tokenizer, ngram_lm_path):
        vocab_dict = tokenizer.get_vocab()
        sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
        vocab = [x[1] for x in sort_vocab][:-1]
        vocab_list = vocab
        # print(vocab_list)
        # print(len(vocab_list), tokenizer.unk_token_id, tokenizer.word_delimiter_token_id)
        # vocab_list[tokenizer.unk_token_id] = ""
        vocab_list[tokenizer.word_delimiter_token_id] = " "
        alphabet = Alphabet.build_alphabet(vocab_list)
        lm_model = kenlm.Model(ngram_lm_path)
        decoder = BeamSearchDecoderCTC(alphabet, language_model=LanguageModel(lm_model))
        return decoder

    # language model
    ngram_lm_model = get_decoder_ngram_model(processor.tokenizer, lm_path)

    def predict(speech, fs):
        # tokenizer
        input_values = processor(speech, sampling_rate=fs, return_tensors='pt', padding='longest').input_values.to(args.device)
        # retrieve logits
        logits = model(input_values).logits
        # argmax
        argmax_prediction = processor.batch_decode(torch.argmax(logits, dim=-1))
        beam_search_output = ngram_lm_model.decode(logits.cpu().detach().numpy().squeeze(), beam_width=500)
        # beam_search_output = ''
        return argmax_prediction[0], beam_search_output


    with open('seg_infer.txt', 'w', encoding='utf8') as fp:
        filename = args.wav
        speech, fs = sf.read(filename)
        assert fs == 16000

        for idx, (start, end) in enumerate(segmentation(filename, min_duration=2, max_duration=10)):
            p, p_lm = predict(speech[int(start * fs):int(end * fs)], fs)
            print(idx+1, start, end, p_lm)
            fp.write(f'{idx+1},{start:.02f},{end:.02f},{p_lm}\n')


if __name__ == '__main__':
    main()
    
