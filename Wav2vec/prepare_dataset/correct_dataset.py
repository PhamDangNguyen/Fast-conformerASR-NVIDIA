import csv
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers.trainer_utils import get_last_checkpoint
import random
import torch
import torch.nn as nn
import numpy as np
import argparse
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
import kenlm
import copy
import soundfile as sf
from statistics import mean
import math
from forced_align import check_data
import pandas as pd
from jiwer import wer

TIME_STEP_WINDOW: float = 0.02
MIN_PROB_WORD = 0.4

list_check = ['mix_train.csv','new_augment_13_11_2023_train.csv','denoise_dns48_train.csv','noise_train.csv','augment_codec_wo_bg_train.csv','Telesale_2_2023_train.csv','music_train.csv','Telesale3_09_2022_train.csv','verified_common_voice_12_13_train.csv','Telesale_2_2023_val.csv','Telesale3_09_2022_val.csv','augment_compression_08_2023_val.csv','augment_compression_08_2023_train.csv','fpt_train.csv']

def recursive_walk(folder):
    for root,dirs,files in os.walk(folder):
        for file_ in files:
            if file_.endswith(".csv"):
                # if file_ in list_check:
                # if file_ == 'mix_train.csv':
                    yield os.path.join(root,file_)

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=str,default='/home/ndanh/asr-wav2vec/output/checkpoint-17120000_13340000', help="Path to checkpoint folder")
    parser.add_argument("--lm", type=str,default='/home/ndanh/asr-wav2vec/ngram_lm/[2022_05]vi_4gram.binary', help="path to lm_path")
    parser.add_argument("--dataset", type=str, default="dataset_cmc_remove_duplicate", help="Path to folder containing csv")
    parser.add_argument("--vocab", default="vocab/vocab_large.json")
    parser.add_argument("--save_path", type=str, default="correct_dataset_cmc", help="Path to folder containg new csv files")
    parser.add_argument("--final_save_path", type=str, default="final_correct_dataset_cmc", help="Path to folder containg new csv files")
    args = parser.parse_args()
    return args

class CMCWav2vec(nn.Module):

    def __init__(self, vocab, last_checkpoint, device='cpu'):
        super().__init__()
        self.processor = Preprocessor(vocab, device)
        self.classifier = Wav2Vec2ForCTC.from_pretrained(last_checkpoint).to(device)
        self.classifier.eval()
        self.device = device
    def forward(self, x):
        
        input_values = self.processor(x)
        
        logits = self.classifier(input_values.to(self.device)).logits
        
        return logits

class Preprocessor(nn.Module):

    def __init__(self, vocab, device='cpu'):
        super().__init__()
        self.tokenizer = Wav2Vec2CTCTokenizer(
            vocab, 
            unk_token='<unk>', 
            pad_token='<pad>', 
            word_delimiter_token=' '
        )
        
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, 
            sampling_rate=16000, 
            padding_value=0.0, 
            do_normalize=True, 
            return_attention_mask=False
        )

        self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)
        self.device = device


    def decode(self, logits):
        argmax_prediction = self.processor.batch_decode(torch.argmax(logits, dim=-1))
        return argmax_prediction[0]

    def forward(self, x):
        input_values = self.processor(x, sampling_rate=16000, return_tensors='pt', padding='longest').input_values.to(self.device)
        return input_values

def load_asr_model(args, device):
    
    pytorch_model = CMCWav2vec(args.vocab,args.checkpoint, device=device)
    return pytorch_model



def lm_model(asr_model,args):
    # beam search
    vocab_dict = asr_model.processor.processor.tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab = [x[1] for x in sort_vocab][:-1]
    vocab_list = vocab
    alphabet = Alphabet.build_alphabet(vocab_list)
    lm_model = kenlm.Model(args.lm)
    decoder = BeamSearchDecoderCTC(alphabet, language_model=LanguageModel(lm_model))
    return decoder

def get_dictionary(asr_model):
    vocab_dict = asr_model.processor.processor.tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab = [x[1] for x in sort_vocab][:-2]
    raw_vocab = vocab
    labels_vocab = copy.deepcopy(raw_vocab)
    labels_vocab[asr_model.processor.processor.tokenizer.word_delimiter_token_id] = ''
    dictionary = {c: i for i, c in enumerate(labels_vocab)}
    return dictionary
# dataset_cmc_remove_duplicate/augment_loud_train.csv
def asr_pred(asr_model,ngram_model,speech,fs,device):
    # tokenizer
    # retrieve logits
    logits = asr_model(speech).detach().cpu()
    logits_tensor = copy.deepcopy(logits.to(torch.float32))
    argmax_prediction = asr_model.processor.processor.batch_decode(torch.argmax(logits, dim=-1)[0])
    beam_search_output = ngram_model.decode_beams(logits.numpy().squeeze(), beam_width=500)[0]
    return argmax_prediction, beam_search_output, torch.log_softmax(logits, dim=-1), logits_tensor

def infer():
    args = parser_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    asr_model = load_asr_model(args,device)
    ngram_model = lm_model(asr_model, args)
    dictionary = get_dictionary(asr_model)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    error = open("error_v2.txt",'w')
    for file_ in recursive_walk(args.dataset):
        print(f"[+]Processing : {file_}")
        csv_reader = open(file_).read().strip().split('\n')[1:]
        base_name = os.path.basename(file_)
        final_results = []
        for row in csv_reader:
            transcript_scores = []
            parts = row.split(",")
            audio_path = parts[0]
            
            audio, sr = sf.read(audio_path, dtype=np.float32)
            
            duration = audio.shape[0] / sr
            try:
                p, p_lm, logits, logits_tensor = asr_pred(asr_model,ngram_model,audio,16000,device)
                # print(logits.shape)
                
                list_align, list_space = check_data(parts[1], audio, dictionary,p,logits)
                softmax_pred = torch.softmax(logits_tensor, dim=-1)
                max_elements, _ = torch.max(softmax_pred, dim=-1)
                max_elements = max_elements.tolist()
                
                
                end_in_time_step_window = math.floor(duration / TIME_STEP_WINDOW)

                all_words = []
                
                for word_index, (word, (start, end)) in enumerate(p_lm[2]):
                    # normalize end, something end word larger than segment duration
                    word = word.lower()
                    end = end if end <= end_in_time_step_window else end_in_time_step_window

                    confident_score_list = max_elements[0][start:end]
                    
                    confident_score = mean(confident_score_list) if len(confident_score_list) > 0 else 0.0
                    if confident_score <= MIN_PROB_WORD:
                        print(confident_score)
                        continue
                    transcript_scores.append(confident_score)
                    all_words.append(word)
                pred_trans = " ".join(word for word in all_words)
                pred_trans = pred_trans.strip()
                
                if len(transcript_scores):
                    final_results.append([parts[0],parts[1],duration,pred_trans,mean(transcript_scores),list_align])
                else:
                    final_results.append([parts[0],parts[1],duration,pred_trans,0.0,list_align])
            except:
                print(f"Error file: {audio_path}")
                error.write(f"{audio_path}\n")


        with open(f"{args.save_path}/{base_name}",'w') as g:
            csv_writer = csv.writer(g)
            csv_writer.writerow(['audio_path','label','duration','pred','confidence','list_align'])
            csv_writer.writerows(final_results)

def get_dataset_info():
    args = parser_args()
    csv_files = recursive_walk(args.final_save_path)
    dataset_info = {}
    for csv_file in csv_files:
        if csv_file.endswith("train.csv"):
            pure_name = os.path.splitext(os.path.basename(csv_file))[0]
            if os.path.exists(csv_file.replace("train.csv","val.csv")):
                all_files = [csv_file,csv_file.replace("train.csv","val.csv")]
            else:
                all_files = [csv_file]
    
            data = pd.concat([pd.read_csv(f) for f in all_files], axis=0, ignore_index=True)
            durations = data['duration'].sum()
            dataset_info[pure_name] = durations / 3600

    print(f"Lengths of each dataset")
    for key, value in dataset_info.items():
        print(f"Dataset: {key} - Lengths (h): {value}")

def remove_hard_sample():
    print("-------------------------------------------------------------")
    args = parser_args()
    if not os.path.exists(args.final_save_path):
        os.makedirs(args.final_save_path, exist_ok=True)
    csv_files = recursive_walk(args.save_path)
    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        print(f"Total number of rows before filter confidence: {len(data)}")

        data = data[data['confidence'] > 0.65]
        
        print(f"Total number of rows after filter confidence: {len(data)}")

        print("Filter high wer")

        max = -10
        new_data = []
        for idx in range(len(data)):
            batch = data.iloc[idx].copy()
            wer_pred = wer(batch['label'],batch['pred'])
            if max < wer_pred:
                max = wer_pred
            
            if wer_pred < 0.4:
                new_data.append(batch)


        print(f"Dataset: {os.path.basename(csv_file)} - Max WER: {wer_pred}")
        data_filter = pd.DataFrame(new_data)
        data_filter.to_csv(os.path.join(args.final_save_path,os.path.basename(csv_file)))

def gen_train_data():
    args = parser_args()
    csv_files = recursive_walk(args.save_path)
    os.makedirs("final_correct_dataset_cmc_filter",exist_ok=True)
    for csv_file in csv_files:
        with open(f"final_correct_dataset_cmc_filter/{os.path.basename(csv_file)}",'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['file','text','duration'])
            df = pd.read_csv(csv_file)
            num_lines = df.shape[0]

            g = open(csv_file)
            csv_reader = csv.reader(g)
            
            fields = next(csv_reader)
            num_keep = num_lines
            if "augment" in csv_file and "train" in csv_file:
                num_keep = math.floor(0.2*num_lines)

            csv_reader = list(csv_reader)
            random.shuffle(csv_reader)
            
            csv_reader = csv_reader[:num_keep]
            new_rows = []
            for i,row in enumerate(list(csv_reader)):
                
                new_rows.append([row[0],row[1],row[2]])
            
            csv_writer.writerows(new_rows)


if __name__ == '__main__':
    # infer()
    # remove_hard_sample()
    get_dataset_info()
    # gen_train_data()