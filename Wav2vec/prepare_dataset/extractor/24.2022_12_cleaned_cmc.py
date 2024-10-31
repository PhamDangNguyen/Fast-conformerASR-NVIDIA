from utils import *

import os
import json
from scipy.io import wavfile
import tqdm
import random

DATASET_PATH = '/media/storage/hai/dataset/24.2022_12_cleaned_cmc'
PREFIX = 'E:\Final12.2022'

OUTPUT_NAME = '2022_12_cleaned_cmc'

def export():
    with open(f'{OUTPUT_NAME}.csv', 'w') as fp:
        fp.write('file,text,duration\n')
        for audio_filepath in tqdm.tqdm([_ for _ in recursive_walk(DATASET_PATH) if _.endswith('.wav')]):
            if '_16k.wav' in audio_filepath:
                continue
                # print(line)
            text = clean_text(open(audio_filepath[:-4] + '.txt').read().strip())
            if has_numbers(text):
                continue
            
            if len(text) == 0:
                print('none text')
                continue
            # print(audio_filepath, text)
            fs, speech = wavfile.read(audio_filepath)
            if fs != 16000:
                new_audio_filepath = audio_filepath[:-4] + '_16k.wav'
                os.system(CMD_CONVERT_1CH.format(audio_filepath, new_audio_filepath))
                audio_filepath = new_audio_filepath

            fs, speech = wavfile.read(audio_filepath)
            assert fs == 16000
            duration = len(speech)/fs
            if duration > 20:
                continue
            fp.write(f'{audio_filepath},{text},{duration}\n')


def split_train_val(num_val=23):
    print('\nSplitting dataset\n')
    lines = open(f'{OUTPUT_NAME}.csv').read().strip().split('\n')[1:]
    random.shuffle(lines)
    with open(f'../dataset/{OUTPUT_NAME}_val.csv', 'w') as fp:
        fp.write('file,text,duration\n')
        for line in lines[:num_val]:
            fp.write(f'{line}\n')

    with open(f'../dataset/{OUTPUT_NAME}_train.csv', 'w') as fp:
        fp.write('file,text,duration\n')
        for line in lines[num_val:]:
            fp.write(f'{line}\n')

    print(len(lines)-num_val, num_val)

export()
split_train_val()