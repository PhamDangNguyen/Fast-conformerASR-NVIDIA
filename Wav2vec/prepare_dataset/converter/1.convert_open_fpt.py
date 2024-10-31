# import soundfile as sf
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


data_dir = '/home/duonghai/dataset_voice/1.open_fpt'

transcript = os.path.join(data_dir, 'transcriptAll.txt')
audio_dir = os.path.join(data_dir, 'mp3')

save_audio_dir = os.path.join(data_dir, 'wav')
# os.system(f'rm -rf {save_audio_dir}')

print(f'Converting {data_dir}...')

if not os.path.exists(save_audio_dir):
    os.makedirs(save_audio_dir)

def f(line):
    name, _, _ = line.split('|')
    i = os.path.join(audio_dir, name)
    o = os.path.join(save_audio_dir, name[:-4] + '.wav')
    if os.path.exists(i):
        if not os.path.exists(o):
            # command = f'ffmpeg -i {i} {o}_tmp.wav'
            # os.system(command)
            # command = f'sox {o}_tmp.wav -r 16000 -c 1 -b 16 {o}'
            # os.system(command)
            # command = f'rm -rf {o}_tmp.wav'
            # os.system(command)

            command = f'sox {i} --rate 16k --bits 16 --channels 1 -e signed-integer {o}'
            os.system(command)
            
lines = open(transcript).read().strip().split('\n')
step = 3200
for i in tqdm.tqdm(range(0, len(lines), step)):
    tmp_lines = lines[i: i + step]
    Parallel(n_jobs=os.cpu_count() // 2)(delayed(f)(line) for line in tmp_lines)
