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
    total = 0
    config_paths = ['../nemo_asr/config/dataset/mix.json']
    for config_path in config_paths:
        count_missing = 0
        count = 0
        print(f'[+] Checking {config_path}')
        for meta in read_config(config_path):
            path = meta['audio_filepath']
            count += 1
            if not os.path.exists(path):
                print(f'    missing {path}')
                count_missing += 1
                total += 1
        print(f'    {config_path} missing {count_missing}/{count} files')

    print(f'[+] Missing {total} files')
