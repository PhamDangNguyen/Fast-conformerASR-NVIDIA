import os
import json
import random
import tqdm
import numpy as np
from scipy.io import wavfile


def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)

def choose(dataset):
    lines = open(dataset).read().strip().split('\n')[1:]
    while True:
        line = random.choice(lines)
        duration = float(line.split(',')[-1])
        if duration < 10:
            return line
    
    
if __name__ == '__main__':
    MAX_FILES = 5000
    FS = 16000
    datasets = [_ for _ in recursive_walk('../dataset') if _.endswith('_val.csv')]
    print(datasets)
    
    OUTPUT_DATA_DIR = '/media/storage/hai/dataset/31.concat_stt_2_persons'
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    
    silence = np.zeros([20 * FS // 1000], dtype=np.int16)
    
    with open(f'concat_stt_2_persons_train.csv', 'w') as fp:
        print(f'file,text,duration', file=fp)
        for i in tqdm.tqdm(range(MAX_FILES)):
            line1 = choose(random.choice(datasets))
            line2 = choose(random.choice(datasets))
            
            f1, t1, _ = line1.split(',')
            f2, t2, _ = line2.split(',')
            
            _, s1 = wavfile.read(f1)
            _, s2 = wavfile.read(f2)
            
            s_new = np.concatenate([s1, silence, s2])
            f_new = os.path.join(OUTPUT_DATA_DIR, f'concat_{os.path.basename(f1[:-4])}_{os.path.basename(f2[:-4])}.wav')
            if os.path.exists(f_new):
                continue
            duration = (len(s1) + len(s2) + 20) / FS
            wavfile.write(f_new, FS, s_new)
            fp.write(f'{f_new},{t1} {t2},{duration}\n')
        
        
        # print(line1)
        # print(line2)
    