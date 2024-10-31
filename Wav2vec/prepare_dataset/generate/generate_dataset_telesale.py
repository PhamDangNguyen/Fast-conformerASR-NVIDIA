"""
Generate dataset TTS + Data gán nhãn dữ liệu: telesale, ytb, cmc tự thu âm
"""
import unicodedata as ud
import random
import json
import tqdm
import os
import re

random.seed(3636)
vocab = []


def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)


def read_config(filename):
    with open(filename) as fid:
        return [json.loads(l) for l in fid]


def generate(path):
    global vocab
    print('processing:', path)

    os.makedirs('dataset', exist_ok=True)

    metas = read_config(path)
    random.shuffle(metas)
    name = os.path.basename(path).split('.')[0]
    with open(f'dataset/{name}_val.csv', 'w', encoding='utf8') as fp:
        fp.write('file,text,duration\n')
        for meta in tqdm.tqdm(metas[:100]):
            file = meta['audio_filepath'].replace('/media/storage/hai/dataset/11.media_cmc/',
                                                  '/home/dvhai/dataset/11.media_cmc/')

            if not os.path.exists(file):
                print(file)
                continue

            duration = meta['duration']
            if duration > 20.0 or duration < 1.0:
                continue

            text = meta['text']
            if re.findall(r'[0-9]+', text):
                continue
            text = re.sub(r'\_+', r' ', text)
            text = re.sub(r'\W+', r' ', text)
            text = re.sub(r' +', r' ', text)
            text = text.strip().lower()

            if text == '':
                print('skip:', file)
                continue

            vocab += list(text)
            fp.write(f'{file},{text},{duration}\n')

    with open(f'dataset/{name}_train.csv', 'w', encoding='utf8') as fp:
        fp.write('file,text,duration\n')
        for meta in tqdm.tqdm(metas[100:]):
            file = meta['audio_filepath'].replace('/media/storage/hai/dataset/11.media_cmc/',
                                                  '/home/dvhai/dataset/11.media_cmc/')

            if not os.path.exists(file):
                continue

            duration = meta['duration']
            if duration > 20.0 or duration < 1.0:
                continue

            text = meta['text']
            if re.findall(r'[0-9]+', text):
                continue
            text = re.sub(r'\_+', r' ', text)
            text = re.sub(r'\W+', r' ', text)
            text = re.sub(r' +', r' ', text)
            text = text.strip().lower()

            if text == '':
                print('skip:', file)
                continue

            vocab += list(text)
            fp.write(f'{file},{text},{duration}\n')

    vocab = list(sorted(set(vocab)))
    print(len(vocab))


if __name__ == '__main__':
    path = '/home/dvhai/dataset/12.cleaned_cmc/media_cmc.json'
    generate(path)

    # vocab = list(sorted(set(vocab)))
    # vocab_dict = {v: k for k, v in enumerate(vocab)}
    # vocab_dict["[UNK]"] = len(vocab_dict)
    # vocab_dict["[PAD]"] = len(vocab_dict)

    # with open('dataset/vocab.json', 'w', encoding='utf8') as fp:
    #     json.dump(vocab_dict, fp, indent=4, ensure_ascii=False)
    #
