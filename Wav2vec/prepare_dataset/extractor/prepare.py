from scipy.io import wavfile
from pprint import pprint
import json
import tqdm
import time
import os

def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)

def read_config(filename):
    with open(filename) as fid:
        return [json.loads(l) for l in fid]

def new_config():

    old_path = '/home/duonghai/dataset_voice/'
    new_path = '/media/storage/hai/dataset/'
    config_path = 'data/train.config'

    correct_wavs = open('data/correct_wav_1_error.txt').read().strip().split('\n')
    
    with open('./config/new_train.config', 'w', encoding='utf8') as fp:
        for meta in tqdm.tqdm(read_config(config_path)):
            path = meta['audio_filepath']
            if path in correct_wavs:
                new_filepath = path.replace(old_path, new_path)
                if not os.path.exists(new_filepath):
                    print(f'missing {new_filepath}')
                    continue

                correct_meta = {
                    'audio_filepath': new_filepath,
                    'duration': meta['duration'],
                    'text': meta['text'],
                }
                json.dump(correct_meta, fp, ensure_ascii=False)
                fp.write('\n')


def new_config_fpt():
    path = '/media/storage/hai/dataset/fpt'
    count = 0
    error = 0

    with open('./config/fpt.config', 'w', encoding='utf8') as fp:
        for file in recursive_walk(path):
            if file.endswith('_new.wav'):
                try:
                    text = open(file[:-8] + '.txt').read().strip()
                    fs, data = wavfile.read(file)
                    duration = float(data.shape[0] / fs)

                    meta = {
                        'audio_filepath': file,
                        'duration': duration,
                        'text': text.lower(),
                    }
                    json.dump(meta, fp, ensure_ascii=False)
                    fp.write('\n')

                except KeyboardInterrupt:
                    raise

                except: 
                    error += 1
                else:
                    count += 1

                    if count % 1000 == 0:
                        print(count, error)


def new_config_youtube():
    dataset_dirs = ['/media/storage/hai/dataset/datasets_18092021/']
    # dataset_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if d.startswith('datasets_')]

    with open('/home/dvhai/code/nemo_asr/config/dataset/youtube_20210918.json', 'w', encoding='utf8') as fp:
        for d in dataset_dirs:
            bar = tqdm.tqdm([f for f in recursive_walk(d) if (f.endswith('.json') and not f.endswith('meta.json'))])
            bar.set_description('dataset: {}'.format(os.path.basename(d)))
            for config_path in bar:
                # print(config_path)
                for meta in read_config(config_path):
                    if '_0.wav' in meta['audio_filepath']:
                        continue
                    if not os.path.exists(meta['audio_filepath']):
                        print('[ERROR] file "{}" does not exist.'.format(meta['audio_filepath']))
                        continue
                    fs, _ = wavfile.read(meta['audio_filepath'])
                    if fs == 16000:
                        json.dump(meta, fp, ensure_ascii=False)
                        fp.write('\n')
                    else:
                        print('[ERROR] file "{}", fs={}'.format(meta['audio_filepath'], fs))

if __name__ == '__main__':
    new_config_youtube()
