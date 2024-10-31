from scipy.io import wavfile
import requests
import json
import tqdm
import os
import re
import random
import time
import argparse
import traceback

from utils.wer import wer

URL = "https://voicestreaming.cmccist.ai/speech_to_text"
# 103.252.1.156:8907

def random_ip():
    return ".".join(str(random.randint(0, 255)) for _ in range(4))

def clean(name, replacement=' '):
    text = re.sub(r'\_+', replacement, name)
    text = re.sub(r'\W+', replacement, text)
    text = re.sub(r' +', replacement, text)
    text = text.strip().lower()
    return text

def cmc_api(path):
    assert os.path.exists(path)
    files = {'content': open(path, 'rb')}
    values = {'is_normalize': 1, 'return_tracking_change_normalize': 1}
    headers={'X-Forwarded-For': random_ip(), 'api_key': 'F5vQXdpQ3K2rbuz8xGMu66bZ691mZDxGV5kF6nQOL1fZDKlpX7N80XmRYj1376TY'}
    # print('requesting', path)
    n_tried = 0
    while True:
        if n_tried > 30:
            return ''
        try:
            r = requests.post(url=URL, files=files, data=values, headers=headers)
            prediction = r.json()["raw_output"]
        except requests.exceptions.ConnectionError:
            print('connection timeout')
            print(traceback.format_exc())
            time.sleep(10)
        except KeyboardInterrupt:
            exit()
        except:
            print(traceback.format_exc())
            print('error', n_tried, path)
            time.sleep(1)
            n_tried += 1
        else:
            break
    # print('done')
    return prediction

def check_oov(vocab, text):
    for char in text:
        if char not in vocab:
            return True
    return False


def export_verified_dataset(csv_name, vocab):
    with open(f'verified_{os.path.basename(csv_name)}', 'w') as fp:
        print(f'file,text,duration', file=fp)
        for line in tqdm.tqdm(open(csv_name).read().strip().split('\n')):
            path, text, prediction = line.split(',')
            assert os.path.exists(path)
            
            if check_oov(vocab, text):
                print('OOV:', text)
                continue
            w = wer([clean(text)], [clean(prediction)], use_tqdm=False)
            if w > 10:
                continue
                
            print('WER:', w)
            print(text)
            print(prediction)
            
            fs, speech = wavfile.read(path)
            duration = len(speech) / fs
            if duration < 1.0 or duration >= 20:
                print('Duration:', duration, text)
                continue

            fp.write(f'{path},{clean(text)},{duration}\n')
        



def export_large_wer_dataset(csv_name, vocab, thresholds):
    with open(f'wer_{thresholds[0]}_{thresholds[1]}_{os.path.basename(csv_name)}', 'w') as fp:
        print(f'file,text,duration', file=fp)
        for line in tqdm.tqdm(open(csv_name).read().strip().split('\n')):
            path, text, prediction = line.split(',')
            assert os.path.exists(path)
            
            if check_oov(vocab, text):
                print('OOV:', text)
                continue
            w = wer([clean(text)], [clean(prediction)], use_tqdm=False)
            if thresholds[0] <= w <= thresholds[1]:
                print('WER:', w)
                print(text)
                print(prediction)
                
                fs, speech = wavfile.read(path)
                duration = len(speech) / fs
                if duration < 1.0 or duration >= 20:
                    print('Duration:', duration, text)
                    continue

                fp.write(f'{path},{clean(text)},{duration}\n')
            
        
        
if __name__ == '__main__':

    # final_tiktok_label_filter_2_1ch_train.csv
    # final_tiktok_label_filter_3_1ch_train.csv
    # tiktok_crawl_orig_1ch_train
    
    csv_name = 'dataset/tiktok_crawl_orig_1ch_train.csv'
    csv_out = f'cmc_prediction_{os.path.basename(csv_name)}'
    
    if 0:    
        pbar = tqdm.tqdm(open(csv_name).read().strip().split('\n')[1:])
        pbar.set_description(csv_name)
        with open(csv_out, 'w') as fp:
            for line in pbar:
                path, text, duration = line.split(',')
                prediction = cmc_api(path)
                if prediction == '':
                    continue
                print(f'{path},{clean(text)},{clean(prediction)}', file=fp)
    
    vocab = list(json.load(open('../wav2vec/vocab/vocab_large.json')).keys())
    
    # text = "y cùng hai ca sĩ fatih erkoç và müslüm gürses"
    # text = "y cùng hai ca sĩ "
    # print(check_oov(vocab, text))
    
    # csv_out = 'cmc_prediction_fleur_train.csv'
    # export_verified_dataset(csv_out, vocab)
    for thresholds in [[10, 20], [20, 30], [30, 40], [40, 50]]:
        export_large_wer_dataset(csv_out, vocab, thresholds=thresholds)