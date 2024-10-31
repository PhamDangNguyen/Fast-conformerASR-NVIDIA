from scipy.io import wavfile
import librosa
import json
import tqdm
import os


data_dir = '/home/duonghai/dataset_voice/6.vivos'
type = 'train'
# ${data_dir}/${type}/prompts.txt
transcript = os.path.join(data_dir, type, 'prompts.txt')
audio_dir = os.path.join(data_dir, type, 'waves')

print(f'Extracting {data_dir}...')

with open(f'config/6.vivos_{type}.config', 'w', encoding='utf8') as fp:
    for line in tqdm.tqdm(open(transcript).read().strip().split('\n')):
        # print('[+]', line) # 2800~chị ẩn cánh cửa bước vào~2.415979166666667
        name = line.split(' ')[0]
        dirname = name.split('_')[0]

        text = line.replace(name + ' ', '').strip().lower().replace('\"', '')

        abs_path = os.path.join(audio_dir, dirname, name + '_new.wav')

        if os.path.exists(abs_path):
            fs, data = wavfile.read(abs_path)

            # print(f'    - name: {name}')
            # print(f'    - text: {text}')
            # print('    - duration: {}'.format(data.shape[0]/fs))
            meta = {
                'audio_filepath': abs_path,
                'duration': data.shape[0]/fs,
                'text': text.lower()
            }
            json.dump(meta, fp, ensure_ascii=False)
            fp.write('\n')
        else:
            print(f'{abs_path} does not exist')

