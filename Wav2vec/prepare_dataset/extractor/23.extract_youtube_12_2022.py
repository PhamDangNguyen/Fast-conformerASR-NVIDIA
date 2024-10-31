from utils import *

import os
import json
from scipy.io import wavfile
import tqdm
import random

DATASET_PATH = '/media/storage/hai/dataset/23.youtube_12_2022'
PREFIX = 'E:\Final12.2022'

def export():
    with open('youtube_12_2022.csv', 'w') as fp:
        fp.write('file,text,duration\n')
        for file in tqdm.tqdm([_ for _ in recursive_walk(DATASET_PATH) if _.endswith('.json')]):
            try:
                lines = read_config(file)
            except:
                print('error:', file)
                continue
            for line in lines:
                # print(line)
                audio_filepath = line['audio_filepath'].replace(PREFIX, DATASET_PATH).replace('\\', '/')
                text = clean_text(line['text'])
                if has_numbers(text):
                    print('has text')
                    continue

                if len(text) == 0:
                    print('none text')
                    continue

                # print(audio_filepath, text)
                # exit()
                fs, speech = wavfile.read(audio_filepath)
                assert fs == 16000
                duration = len(speech)/fs
                if duration > 20:
                    continue
                fp.write(f'{audio_filepath},{text},{duration}\n')


def split_train_val(num_val=200):
    print('\nSplitting dataset\n')
    lines = open('youtube_12_2022.csv').read().strip().split('\n')[1:]
    random.shuffle(lines)
    with open('../dataset/youtube_12_2022_val.csv', 'w') as fp:
        fp.write('file,text,duration\n')
        for line in lines[:num_val]:
            fp.write(f'{line}\n')

    with open('../dataset/youtube_12_2022_train.csv', 'w') as fp:
        fp.write('file,text,duration\n')
        for line in lines[num_val:]:
            fp.write(f'{line}\n')

    print(len(lines)-num_val, num_val)

export()
split_train_val()