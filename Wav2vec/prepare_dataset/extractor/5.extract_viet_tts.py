from scipy.io import wavfile
import librosa
import json
import tqdm
import os


data_dir = '/home/duonghai/dataset_voice/5.viet_tts'

transcript = os.path.join(data_dir, 'meta_data.tsv')
audio_dir = os.path.join(data_dir, 'wav')

print(f'Extracting {data_dir}...')

with open('config/5.viet_tts.config', 'w', encoding='utf8') as fp:
    for line in tqdm.tqdm(open(transcript).read().strip().split('\n')):
        # print('[+]', line) # 2800~chị ẩn cánh cửa bước vào~2.415979166666667
        name, text = line.split('\t')
        name = os.path.basename(name)[:-4]

        text = text.replace("\"", "")
        abs_path = os.path.join(audio_dir, name + '_new.wav')
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


# sox /home/duonghai/dataset_voice/5.viet_tts/wav/000000.wav -r 16000 -c 1 -b 16 test.wav