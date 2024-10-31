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


data_dir = '/home/duonghai/dataset_voice/9.rdc/ok'

print(f'Converting {data_dir}...')

def f(file):
    name = file[:-4]
    i = name + '.mp3'
    o = name + '_new.wav'
    # print()
    # print(i)
    # print(o)
    # exit()
    if os.path.exists(i):
        if not os.path.exists(o):
            command = f'sox "{i}" --rate 16k --bits 16 --channels 1 -e signed-integer "{o}"'
            os.system(command)
    else:
        print(f'{i} does not exist')
            
# lines = open(transcript).read().strip().split('\n')
# step = 3200
# for i in tqdm.tqdm(range(0, len(lines), step)):
#     tmp_lines = lines[i: i + step]
#     Parallel(n_jobs=os.cpu_count() // 2)(delayed(f)(line) for line in tmp_lines)

files = [_ for _ in recursive_walk(data_dir) if _.endswith('.mp3')]

for file in tqdm.tqdm(files):
    f(file)
