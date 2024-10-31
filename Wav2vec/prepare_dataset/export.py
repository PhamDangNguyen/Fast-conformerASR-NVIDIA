from scipy.io import wavfile
import numpy as np
import unicodedata
import json
import tqdm
import time
import os
import re
import random


def has_numbers(text):
    return any(c.isdigit() for c in text)


def clean_name(name, replacement=''):
    text = os.path.basename(name)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
    text = text.replace('wav', '')
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


def load_json(path):
    with open(os.path.join(path), encoding='utf-8') as f:
        data = json.load(f)
    return data


def build_audio2path(raw_audio_folder, ext='.wav'):
    return {clean_name(f): f for f in recursive_walk(raw_audio_folder) if f.endswith(ext)}


def export_csv(output_csv, json_label_path, raw_audio_folder, date, ext='.wav'):
    audio2path_dict = build_audio2path(raw_audio_folder, ext=ext)

    data = load_json(json_label_path)
    pbar = tqdm.tqdm(data)
    error_number = error_text = error_duration = 0
    count_all = 0
    with open(output_csv, 'w') as fp:
        fp.write('file,text,duration\n')
        for meta in pbar:
            audio_name = meta['original_filename']
            pbar.set_description(audio_name)
            # print(audio2path_dict[clean_name(audio_name)], audio_name)
            # exit()
            # continue
            audio_path = audio2path_dict[clean_name(audio_name)]

            if not os.path.exists(audio_path):
                print('missing:', audio_name)
                error += 1
                continue

            new_filename = 'tmp.wav'

            cmd = f'ffmpeg -y -i "{audio_path}" -ac 1 -ar 16000 "{new_filename}" > /dev/null 2>&1 < /dev/null'
            os.system(cmd)

            fs, speech = wavfile.read(new_filename)
            count = 0
            dirname = os.path.join(os.path.dirname(audio_path), f'splits_{date}')
            # os.system(f'rm -rf {dirname}')
            os.makedirs(dirname, exist_ok=True)

            for seg in meta['segmentations']:
                start = float(seg['start_time'])
                end = float(seg['end_time'])
                text = seg['transcription']
                if not text:
                    # print('error:', audio_name)
                    error_text += 1
                    continue

                if has_numbers(text):
                    error_number += 1
                    continue

                if end - start > 20:
                    error_duration += 1
                    continue

                text = re.sub(r'\_+', ' ', text)
                text = re.sub(r'\W+', ' ', text)
                text = re.sub(r' +', ' ', text)
                text = text.strip().lower()

                new_path = os.path.join(dirname, f'{audio_name[:-4]}_{count}.wav')
                wavfile.write(new_path, fs, speech[int(start * fs): int(end * fs)])
                count += 1
                count_all += 1
                pbar.set_postfix(count=count, total=len(meta['segmentations']), error_text=error_text,
                                 error_number=error_number, error_duration=error_duration, all=count_all)

                fp.write(f'{new_path},{text},{end - start:.05f}\n')


def split_train_val(csv_name, num_files=300):
    #     # normal files
    random.seed(1)
    orig_lines = open(csv_name).read().strip().split('\n')[1:]
    random.shuffle(orig_lines)
    train_lines = orig_lines[:-num_files]
    val_lines = orig_lines[-num_files:]

    with open(f'{csv_name[:-4]}_train.csv', 'w') as fp:
        fp.write('file,text,duration\n')

        for line in tqdm.tqdm(train_lines):
            filename, text, duration = line.split(',')
            assert os.path.exists(filename)
            assert len(text) > 0
            assert float(duration) <= 20
            print(line, file=fp)

    with open(f'{csv_name[:-4]}_val.csv', 'w') as fp:
        fp.write('file,text,duration\n')

        for line in tqdm.tqdm(val_lines):
            filename, text, duration = line.split(',')
            assert os.path.exists(filename)
            assert len(text) > 0
            assert float(duration) <= 20
            print(line, file=fp)


#     # concat files
#     concat_lines = open(f'{output_dir}/ptv_{category}_concat.csv').read().strip().split('\n')
#     for line in concat_lines:
#         filename, text = line.split('|')
#         assert os.path.exists(filename)
#         assert len(text) > 0
#         fp.write(f'{filename}|{speaker}|{text}\n')


if __name__ == '__main__':

    output_dir = f'outputs/'
    dataset_dir = '/media/storage/hai/dataset'

    date = '20230213'
    output_csv = f'{output_dir}/Telesale_2_2023.csv'
    json_label_path = 'Telesale_2_2023.json'
    raw_audio_folder = f'{dataset_dir}/29.Telesale_02_2023'
    # raw_audio_folder = f'{dataset_dir}/20.telesale_09_22'
    ext = '.wav'

    # clear
    # os.system(f'rm -rf {output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(output_csv):
        export_csv(output_csv, json_label_path, raw_audio_folder, date, ext=ext)
        split_train_val(output_csv)
