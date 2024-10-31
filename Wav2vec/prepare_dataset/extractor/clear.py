import tqdm
import json
import os

def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)

def read_config(filename):
    with open(filename) as fid:
        return [json.loads(l) for l in fid]

if __name__ == '__main__':
    root_dir = '/media/storage/hai/dataset/'
    config_path = '../nemo_asr/config/dataset/mix.json'

    wavs = []
    folders = []

    for meta in tqdm.tqdm(read_config(config_path)):
        wavs.append(meta['audio_filepath'])
        folder = meta['audio_filepath'].replace(root_dir, '').split('/')[0]
        if folder not in folders:
            folders.append(folder)

    count = 0
    for folder in folders:
        print(folder)
        temp_files = [f for f in  recursive_walk(os.path.join(root_dir, folder)) if f.endswith('.wav') or f.endswith('.mp3')]
        for file in tqdm.tqdm(temp_files):
            if file in wavs:
                count += 1
            else:
                cmd = f'rm -rf "{file}"'
                os.system(cmd)

    print(len(wavs), count)
