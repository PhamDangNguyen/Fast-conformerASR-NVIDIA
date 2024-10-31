from scipy.io import wavfile
import tqdm
import json
import copy
import os
import re
import unicodedata


def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)


def clean_name(name, replacement=''):
    text = os.path.basename(name)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
    text = text.replace('mp3', '')
    text = re.sub(r'\_+', replacement, text)
    text = re.sub(r'\W+', replacement, text)
    text = re.sub(r' +', replacement, text)
    text = text.strip().lower()
    return text


def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)


def ptv():
    AUDIO_FOLDER = '/media/storage/hai/dataset/18.TTS_pham_nguyen_son_tung'
    AUDIO_PATHS = {clean_name(f): f for f in recursive_walk(AUDIO_FOLDER) if f.endswith('.mp3')}
    JSON_PATH = '/home/dvhai/ptv_pham_nguyen_son_tung.json'

    with open(os.path.join(JSON_PATH), encoding='utf-8') as f:
        data = json.load(f)
    print(len(data))

    print(AUDIO_PATHS)

    with open('PTV_2.csv', 'w') as fp:
        pbar = tqdm.tqdm(data)
        for meta in pbar:
            if 'user_5' not in meta['assigned_user']['username']:
                continue
            audio_name = meta['original_filename']
            pbar.set_description(audio_name)
            audio_path = AUDIO_PATHS[clean_name(audio_name)]
            # print(audio_name, audio_path)
            new_filename = 'tmp.wav'
            # continue
            cmd = f'ffmpeg -y -i "{audio_path}" -ac 1 -ar 16000 "{new_filename}" > /dev/null 2>&1 < /dev/null'
            os.system(cmd)

            fs, speech = wavfile.read(new_filename)
            count = 0
            for seg in meta['segmentations']:
                start = float(seg['start_time'])
                end = float(seg['end_time'])
                text = seg['transcription']
                if not text:
                    print('error:', audio_name)
                    continue
                text = re.sub(r'\_+', ' ', text)
                text = re.sub(r'\W+', ' ', text)
                text = re.sub(r' +', ' ', text)
                text = text.strip().lower()
                dirname = os.path.join(os.path.dirname(audio_path), 'splits')
                os.makedirs(dirname, exist_ok=True)
                new_path = os.path.join(dirname, f'{audio_name}_{count}.wav')
                wavfile.write(new_path, fs, speech[int(start * fs): int(end * fs)])
                count += 1
                pbar.set_postfix(count=count, total=len(meta['segmentations']))

                fp.write(f'{new_path}|{text}\n')


def telesale():
    AUDIO_FOLDER = '/media/storage/hai/dataset/20.telesale_09_22/'
    AUDIO_PATHS = {clean_name(f): f for f in recursive_walk(AUDIO_FOLDER) if f.endswith('.mp3')}
    JSON_PATH = '/home/dvhai/Telesale3_09_2022.json'

    ALL_WAV_FILES = [f for f in recursive_walk(AUDIO_FOLDER) if f.endswith('.wav')]
    # print(ALL_WAV_FILES[:10])
    ALL_WAV_DICT = {os.path.basename(f): f for f in ALL_WAV_FILES}

    with open(os.path.join(JSON_PATH), encoding='utf-8') as f:
        data = json.load(f)
    print(len(data))

    with open('Telesale3_09_2022.csv', 'w') as fp:
        pbar = tqdm.tqdm(data)
        for meta in pbar:
            # print(meta)
            audio_name = meta['original_filename']
            if audio_name not in ALL_WAV_DICT:
                print('missing:', audio_name)
                continue
            filename = ALL_WAV_DICT[audio_name]

            fs, speech = wavfile.read(filename)
            count = 0
            # print(filename)
            for seg in meta['segmentations']:
                start = float(seg['start_time'])
                end = float(seg['end_time'])
                text = seg['transcription']
                if not text:
                    print('error:', audio_name)
                    continue
                text = re.sub(r'\_+', ' ', text)
                text = re.sub(r'\W+', ' ', text)
                text = re.sub(r' +', ' ', text)
                text = text.strip().lower()
                new_path = f'{filename[:-4]}_split_{count}.wav'
                wavfile.write(new_path, fs, speech[int(start * fs): int(end * fs)])
                count += 1
                pbar.set_postfix(count=count, total=len(meta['segmentations']))
                # print(new_path, start, end, text)
                fp.write(f'{new_path}|{text}\n')
            # break


def ptv_predict():
    JSON_PATH = '/home/dvhai/ptv_pham_nguyen_son_tung.json'
    with open(os.path.join(JSON_PATH), encoding='utf-8') as f:
        data = json.load(f)

    PREDICT_PATH = '/home/dvhai/code/wav2vec2/PTV_2_predict.csv'
    PREDICT_DICT = {}
    for l in [line.split('|') for line in open(PREDICT_PATH).read().strip().split('\n')]:
        # print(l[1], '======', l[2])
        PREDICT_DICT[l[1]] = l[2]

    stat = []
    with open('PTV_2_stat.json', 'w') as fp:
        pbar = tqdm.tqdm(data)
        for meta in pbar:
            if 'user_5' not in meta['assigned_user']['username']:
                continue
            audio_name = meta['original_filename']
            # pbar.set_description(audio_name)
            # audio_path = AUDIO_PATHS[clean_name(audio_name)]
            for seg in meta['segmentations']:
                new_meta = {}
                text = seg['transcription']
                if not text:
                    print('error')
                    continue
                text = re.sub(r'\_+', ' ', text)
                text = re.sub(r'\W+', ' ', text)
                text = re.sub(r' +', ' ', text)
                text = text.strip().lower()
                # seg['prediction'] = 
                if text == PREDICT_DICT[text]:
                    continue
                new_meta['original_filename'] = audio_name
                new_meta['transcription'] = seg['transcription']
                new_meta['normalized'] = text
                new_meta['start'] = seg['start_time']
                new_meta['end'] = seg['end_time']
                new_meta['prediction'] = PREDICT_DICT[text]
                stat.append(new_meta)
            #     break
            # break

        json.dump(stat, fp, indent=4, ensure_ascii=False)


def telesale_predict():
    JSON_PATH = '/home/dvhai/Telesale3_09_2022.json'
    with open(os.path.join(JSON_PATH), encoding='utf-8') as f:
        data = json.load(f)

    PREDICT_PATH = '/home/dvhai/code/wav2vec2/Telesale3_09_2022_predict.csv'
    PREDICT_DICT = {}
    for l in [line.split('|') for line in open(PREDICT_PATH).read().strip().split('\n')]:
        # print(l[1], '======', l[2])
        PREDICT_DICT[l[1]] = l[2]

    stat = {}
    with open('Telesale3_09_2022_stat.json', 'w') as fp:
        pbar = tqdm.tqdm(data)
        for meta in pbar:
            # if 'user_5' not in meta['assigned_user']['username']:
            #     continue
            user = meta['assigned_user']['username']
            if user not in stat:
                stat[user] = []

            audio_name = meta['original_filename']
            # pbar.set_description(audio_name)
            # audio_path = AUDIO_PATHS[clean_name(audio_name)]
            for seg in meta['segmentations']:
                new_meta = {}
                text = seg['transcription']
                if not text:
                    print('error')
                    continue
                if len(text.split(' ')) < 5:
                    continue

                text = re.sub(r'\_+', ' ', text)
                text = re.sub(r'\W+', ' ', text)
                text = re.sub(r' +', ' ', text)
                text = text.strip().lower()
                # seg['prediction'] = 
                if text == PREDICT_DICT[text]:
                    continue
                new_meta['original_filename'] = audio_name
                new_meta['transcription'] = seg['transcription']
                new_meta['normalized'] = text
                new_meta['start'] = seg['start_time']
                new_meta['end'] = seg['end_time']
                new_meta['prediction'] = PREDICT_DICT[text]
                stat[user].append(new_meta)
            #     break
            # break

        json.dump(stat, fp, indent=4, ensure_ascii=False)


#

ptv_predict()
# telesale_predict()
