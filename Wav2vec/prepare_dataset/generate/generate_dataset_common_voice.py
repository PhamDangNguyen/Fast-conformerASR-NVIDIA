import unicodedata as ud
import random
import json
import tqdm
import os
import re
from scipy.io import wavfile

CMD_CONVERT_1CH = 'ffmpeg -y -i "{}" -ac 1 -ar 16000 "{}" > /dev/null 2>&1 < /dev/null'

def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)


# def generate(path):
    # print('processing:', path)

    # os.makedirs('test', exist_ok=True)

    # metas = read_config(path)
    # name = os.path.basename(path).split('.')[0]
    # with open(f'test/{name}.csv', 'w', encoding='utf8') as fp:
    #     fp.write('file,text,duration\n')
    #     for meta in tqdm.tqdm(metas):
    #         file = meta['audio_filepath']

    #         if not os.path.exists(file):
    #             print(file)
    #             continue

    #         duration = meta['duration']
    #         if duration < 1.0:
    #             print('skip:', file)
    #             continue

    #         text = meta['text']
    #         if re.findall(r'[0-9]+', text):
    #             continue
    #         text = re.sub(r'\_+', r' ', text)
    #         text = re.sub(r'\W+', r' ', text)
    #         text = re.sub(r' +', r' ', text)
    #         text = text.strip().lower()

    #         if text == '':
    #             print('skip:', file)
    #             continue

    #         fp.write(f'{file},{text},{duration}\n')


def generate(paths, csv_name):
    with open(csv_name, 'w', encoding='utf8') as fp:
        print(f'file,text,duration', file=fp)
        for path in paths:
            skip = 0
            print(path)
            rootdir = os.path.join(os.path.dirname(path), 'clips')
            pbar = tqdm.tqdm(open(path).read().strip().split('\n')[1:])
            for line in pbar:
                pbar.set_postfix(skip=skip)
                parts = line.split('\t')
                # assert len(parts) == 10, print(line)
                # continue
                file = os.path.join(rootdir, parts[1])
                text = parts[2]
                
                if re.findall(r'[0-9]+', text):
                    skip += 1
                    continue
                text = re.sub(r'\_+', r' ', text)
                text = re.sub(r'\W+', r' ', text)
                text = re.sub(r' +', r' ', text)
                text = text.strip().lower()
                
                if len(text) < 10:
                    skip += 1
                    continue
        
                if not file.endswith('.mp3'):
                    skip += 1
                    continue
                new_name = file[:-4] + '.wav'
                os.system(CMD_CONVERT_1CH.format(file, new_name))
                fs, speech = wavfile.read(new_name)
                duration = len(speech) / fs
                if duration < 1.0:
                    skip += 1
                    continue
                
                fp.write(f'{new_name},{text},{duration}\n')

                


if __name__ == '__main__':
    root = '/media/storage/hai/dataset/common_voice_12_13/'
    csv_out = './dataset/common_voice_12_13_train.csv'
    
    labels = [_ for _ in recursive_walk(root) if (_.endswith('.tsv') and ('reported' not in _))]
    
    generate(labels, csv_out)
    # print(labels)
    
    # 