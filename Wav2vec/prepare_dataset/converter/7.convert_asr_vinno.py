from joblib import Parallel, delayed
from scipy.io import wavfile
import librosa
import json
import tqdm
import time
import os

def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)


data_dir = '/home/duonghai/dataset_voice/7.asr_vinno'

print(f'Converting {data_dir}...')

def f(file):
    name = os.path.basename(file)[:-4]
    i = os.path.join(data_dir, name + '.wav')
    o = os.path.join(data_dir, name + '_new.wav')
    if os.path.exists(i):
        if not os.path.exists(o):
            command = f'sox {i} --rate 16k --bits 16 --channels 1 -e signed-integer {o}'
            os.system(command)
    else:
        print(f'{i} does not exist')
            
# lines = open(transcript).read().strip().split('\n')
# step = 3200
# for i in tqdm.tqdm(range(0, len(lines), step)):
#     tmp_lines = lines[i: i + step]
#     Parallel(n_jobs=os.cpu_count() // 2)(delayed(f)(line) for line in tmp_lines)

for file in tqdm.tqdm(os.listdir(data_dir)):
    if file.endswith('.wav'):
        f(file)