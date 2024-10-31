from scipy.io import wavfile
import librosa
import json
import tqdm
import os


data_dir = '/home/duonghai/dataset_voice/3.tts_small_vinno'

transcript = os.path.join(data_dir, 'transcript.txt')
audio_dir = os.path.join(data_dir, 'wav')

print(f'Extracting {data_dir}...')

with open('config/3.tts_small_vinno.config', 'w', encoding='utf8') as fp:
    for line in tqdm.tqdm(open(transcript).read().strip().split('\n')):
        # print('[+]', line) # 2800~chị ẩn cánh cửa bước vào~2.415979166666667
        name, text, duration = line.split('~')
        abs_path = os.path.join(audio_dir, name + '_new.wav')
        if os.path.exists(abs_path):
            fs, data = wavfile.read(abs_path)

            # print(f'    - name: {name}')
            # print(f'    - text: {text}')
            # print(f'    - duration: {duration}')
            # print('    - duration: {}'.format(data.shape[0]/fs))
            meta = {
                'audio_filepath': abs_path,
                'duration': data.shape[0]/fs,
                'text': text.lower()
            }
            json.dump(meta, fp, ensure_ascii=False)
            fp.write('\n')
            # break
        else:
            print(f'{abs_path} does not exist')

        if abs(float(duration) - data.shape[0]/fs) > 1e-2:
            print(float(duration), data.shape[0]/fs)
            print(line)
            exit()
        # exit()
