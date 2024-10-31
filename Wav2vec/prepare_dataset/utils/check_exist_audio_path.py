"""
Check whether audio path in csv file is existed in server or not
"""
import torchaudio
import random
import tqdm
import os

with open('augment_compression_08_2023_train.csv', 'w') as fp:
    print('file,text,duration', file=fp)
    for file in os.listdir(''):
        if not file.endswith('.csv'):
            continue

        if not file.startswith('new_augment'):
            continue

        print('Checking:', file)
        lines = open(file).read().strip().split('\n')
        random.shuffle(lines)
        for line in tqdm.tqdm(lines):
            path, _, duration = line.split(',')

            if not path.endswith('.wav'):
                continue

            if not os.path.exists(path):
                print('Missing:', path)
                continue

            data, fs = torchaudio.load(path)
            new_duration = data.shape[1] / fs
            if abs(new_duration - float(duration)) > 0.1:
                print('Skip:', path, new_duration, float(duration))
                continue
                # raise

            print(line, file=fp)
