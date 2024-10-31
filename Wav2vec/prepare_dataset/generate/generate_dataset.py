import unicodedata as ud
import random
import json
import tqdm
import os
import re

random.seed(3636)
# skip_files = json.loads(open('/home/dvhai/code/nemo_asr/config/clean_dataset_2021_09_29_0.2.manifest').read())
skip_files = {}

def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)

def read_config(filename):
    with open(filename) as fid:
        return [json.loads(l) for l in fid]

def generate(path):
    print('processing:', path)
    os.makedirs('dataset', exist_ok=True)

    metas = read_config(path)
    random.shuffle(metas)
    name = os.path.basename(path).split('.')[0]

    all_samples = len(metas)
    n_val_samples = min(500, int(0.9 * all_samples))

    with open(f'dataset/{name}_val.csv', 'w', encoding='utf8') as fp:
        fp.write('file,text,duration\n')
        for meta in tqdm.tqdm(metas[:n_val_samples]):
            file = meta['audio_filepath']

            if not os.path.exists(file):
                continue

            duration = meta['duration']
            if duration > 20.0 or duration < 1.0:
                continue

            # flag = skip_files.get(file, True)
            # if not flag:
            #     # print('skip:', flag, file)
            #     continue

            text = meta['text']
            if re.findall(r'[0-9]+', text) or len(text) <= 5:
                continue
            text = re.sub(r'\_+', r' ', text)
            text = re.sub(r'\W+', r' ', text)
            text = re.sub(r' +', r' ', text)
            text = text.strip().lower()

            if text == '':
                print('skip:', file)
                continue

            fp.write(f'{file},{text},{duration}\n')


    with open(f'dataset/{name}_train.csv', 'w', encoding='utf8') as fp:
        fp.write('file,text,duration\n')
        for meta in tqdm.tqdm(metas[n_val_samples:]):
            file = meta['audio_filepath']

            if not os.path.exists(file):
                continue

            duration = meta['duration']
            if duration > 20.0 or duration < 1.0:
                continue


            # flag = skip_files.get(file, True)
            # if not flag:
            #     # print('skip:', flag, file)
            #     continue

            text = meta['text']
            if re.findall(r'[0-9]+', text) or len(text) <= 5:
                continue
            text = re.sub(r'\_+', r' ', text)
            text = re.sub(r'\W+', r' ', text)
            text = re.sub(r' +', r' ', text)
            text = text.strip().lower()

            if text == '':
                print('skip:', file)
                continue

            fp.write(f'{file},{text},{duration}\n')
    


if __name__ == '__main__':
    config_dir = '../config'
    for path in recursive_walk(config_dir):
        if 'm4a' in path:
            generate(path)
    
    