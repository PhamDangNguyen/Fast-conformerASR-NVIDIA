import torch
import torch.nn as nn
import argparse
from transformers.trainer_utils import get_last_checkpoint
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
import soundfile as sf
from statistics import mean
import argparse
import os
import csv
import numpy as np
import kenlm
import math
import copy

TIME_STEP_WINDOW: float = 0.02


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=str,default='/home/ndanh/asr-wav2vec/output/checkpoint-19860000', help="Path to checkpoint folder")
    parser.add_argument("--lm", type=str,default='/home/ndanh/asr-wav2vec/ngram_lm/[2022_12_11]vi_4gram_27gb.binary', help="path to lm_path")
    parser.add_argument("--dataset", type=str,default='/home/ndanh/Dialect_classification/huggingface/data/final_train.csv', help="path to csv file path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model and tokenizer
    tokenizer = Wav2Vec2CTCTokenizer(
        f'/home/ndanh/asr-wav2vec/vocab/vocab_large.json', 
        unk_token='<unk>', 
        pad_token='<pad>', 
        word_delimiter_token=' '
    )
    
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, 
        sampling_rate=16000, 
        padding_value=0.0, 
        do_normalize=True, 
        return_attention_mask=False
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    # last_checkpoint = get_last_checkpoint(save_dir)
    # print(f"last_checkpoint: {last_checkpoint}")
    
    # last_checkpoints = ['checkpoint-11660000','checkpoint-11680000','checkpoint-11700000','checkpoint-11720000']
    last_checkpoint = args.checkpoint
    # for last_checkpoint in last_checkpoints:
    
    model = Wav2Vec2ForCTC.from_pretrained(last_checkpoint).to(device)
    model.eval()

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
    
    ngram_lm_model = get_decoder_ngram_model(processor.tokenizer, args.lm)

    def predict(speech,fs,return_logits=False):
        # tokenizer
        input_values = processor(speech, sampling_rate=fs, return_tensors='pt', padding='longest').input_values.to(device)
        # retrieve logits
        logits = model(input_values).logits.detach().cpu()
        logits_tensor = copy.deepcopy(logits.to(torch.float32))
        # argmax
        argmax_prediction = processor.batch_decode(torch.argmax(logits, dim=-1))
        beam_search_output = ngram_lm_model.decode_beams(logits.numpy().squeeze(), beam_width=500)[0]
        # beam_search_output = ''
        
            
        return argmax_prediction[0], beam_search_output, logits, logits_tensor
    
    csv_reader = open(args.dataset).read().strip().split('\n')[1:]
    final_results = []
    
    for row in csv_reader:
        transcript_scores = []
        parts = row.split(",")
        audio_path = parts[0]
        audio, sr = sf.read(audio_path, dtype=np.float32)
        
        duration = audio.shape[0] / sr
        p, p_lm, logits, logits_tensor = predict(audio, 16000)
        softmax_pred = torch.softmax(logits_tensor, dim=-1)
        max_elements, _ = torch.max(softmax_pred, dim=-1)
        max_elements = max_elements.tolist()
        
        end_in_time_step_window = math.floor(duration / TIME_STEP_WINDOW)

        
        for word_index, (word, (start, end)) in enumerate(p_lm[2]):
            # normalize end, something end word larger than segment duration
            word = word.lower()
            end = end if end <= end_in_time_step_window else end_in_time_step_window

            confident_score_list = max_elements[start:end]
            confident_score = mean(confident_score_list[0]) if len(confident_score_list) > 0 else 0.0
            transcript_scores.append(confident_score)
        
        if len(transcript_scores):
            final_results.append([parts[0],parts[1],mean(transcript_scores)])
        else:
            final_results.append([parts[0],parts[1],0.0])

        

    with open("/home/ndanh/Dialect_classification/huggingface/data/preprocess_data_with_api_train.csv",'w') as g:
        csv_writer = csv.writer(g)
        csv_writer.writerow(['audio_path','label','confidence'])
        csv_writer.writerows(final_results)