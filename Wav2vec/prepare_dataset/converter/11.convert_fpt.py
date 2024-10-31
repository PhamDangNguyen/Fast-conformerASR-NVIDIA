from joblib import Parallel, delayed
from scipy.io import wavfile
import json
import tqdm
import time
import os

def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)


def convert(file):
    i = file
    o = file[:-4] + '_new.wav'
    # cmd = f'ffmpeg -y -i "{i}" -ac 1 -ar 16000 "{o}"'
    cmd = f'ffmpeg -y -i "{i}" -ac 1 -ar 16000 "{o}" > /dev/null 2>&1 < /dev/null'
    os.system(cmd)
        
if __name__ == '__main__':
    
    DATA_DIR = '/media/storage/hai/dataset/fpt'
    files = [file for file in recursive_walk(DATA_DIR) if (file.endswith('.wav') and ('_new.wav' not in file))]

    bulk_size = 20
    bulk = []
    for file in tqdm.tqdm(files):

        i = file
        o = file[:-4] + '_new.wav'
        if os.path.exists(o) or ('_new' in i):
            continue

        bulk.append(file)

        if len(bulk) >= bulk_size:
            # for file in bulk:
            #     convert(file)
            Parallel(n_jobs=20)(delayed(convert)(file) for file in bulk)
            bulk = []
