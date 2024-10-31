import requests
import csv
import os
import soundfile as sf
import glob
import json
import ast
from wer import wer
import tqdm
from scipy.io import wavfile
import random


url = "https://voicestreaming.cmccist.ai/speech_to_text/"


payload = {'is_normalize': '0',
'detail_word': '0'}

headers = {
  'api_key': 'mtqijSUcj3hC96vWB6bsmqkgTud7y1tYQ4Tawa0R2V8OwShEAp8E3GEuCZ4F8Uo5'
}

def get_pred_from_api():
    f = open("nghiem_thu_youtube_2_2022_api_pred.csv",'w',encoding='utf8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['filename','label','predict'])

    new_rows = []
    with open("data_cmc_vad_test.csv") as f:
        csv_reader = csv.reader(f)
        fields = next(csv_reader)
        for row in csv_reader:
            audio_path = row[0]
            label = row[1]
            audio, sr = sf.read(audio_path)
            if audio.shape[0]/sr > 30:
                continue
            elif audio.shape[0]/sr <= 0.5:
                new_rows.append([audio_path,label,label])
            else:
                files=[
                ('content',(f'{os.path.basename(audio_path)}',open(f'{audio_path}','rb'),'audio/wav'))
                ]

                response = requests.request("POST", url, headers=headers, data=payload, files=files)
                print(response.text)
                # response = requests.request("POST", url,data=payload, files=files)
                result = ast.literal_eval(response.text)
                new_rows.append([audio_path,label,result["response"]])
    csv_writer.writerows(new_rows)

def get_wer_from_pred():
    count_valid = 0
    coutn_invalid = 0
    bar = tqdm.tqdm(open('nghiem_thu_youtube_2_2022_api_pred.csv').read().strip().split('\n')[1:])
    with open('nghiem_thu_youtube_2_2022.csv', 'w') as f1, open('nghiem_thu_youtube_2_2022_invalid.csv', 'w') as f2:
        f1.write('file,text,duration\n')
        f2.write('file,text,duration\n')
        for line in bar:
            bar.set_postfix(valid=count_valid, invalid=coutn_invalid)
            path, label, predict = line.split(',')
            # print(path, label, predict)
            try:
                w = wer([label], [predict], use_tqdm=False)
            except:
                coutn_invalid += 1
                continue
            # print(w)
            fs, speech = wavfile.read(path)
            duration = len(speech) / fs
            if w <= 8:
                count_valid += 1
                f1.write(f'{path},{label},{duration}\n')
            else:
                coutn_invalid += 1
                f2.write(f'{path},{label},{duration}\n')

def split_train_val():
    bar = open('nghiem_thu_youtube_2_2022.csv').read().strip().split('\n')[1:]
    random.shuffle(bar)
    split = int(0.2 * len(bar))
    dev_split = bar[:split]
    train_split = bar[split:]
    with open('/home/ndanh/asr-wav2vec/dataset_cmc/nghiem_thu_ytb_2022_02_train.csv','w') as f1, open('/home/ndanh/asr-wav2vec/dataset_cmc/nghiem_thu_ytb_2022_02_val.csv','w') as f2:
        f1.write('file,text,duration\n')
        f2.write('file,text,duration\n')
        for line in train_split:
            path, label, duration = line.split(',')
            f1.write(f'{path},{label},{duration}\n')
        
        for line in dev_split:
            path, label, duration = line.split(',')
            f2.write(f'{path},{label},{duration}\n')

if __name__ == '__main__':
    # get_pred_from_api()
    # get_wer_from_pred()
    split_train_val()