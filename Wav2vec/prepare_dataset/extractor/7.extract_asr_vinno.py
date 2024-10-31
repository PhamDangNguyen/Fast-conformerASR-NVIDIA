from scipy.io import wavfile
import librosa
import json
import tqdm
import os


data_dir = '/home/duonghai/dataset_voice/7.asr_vinno'
print(f'Extracting {data_dir}...')

with open(f'config/7.asr_vinno.config', 'w', encoding='utf8') as fp:
    for file in tqdm.tqdm(os.listdir(data_dir)):
        if file.endswith('.wav') and ('_new.wav' not in file):
            name = os.path.basename(file)[:-4]
            abs_path = os.path.join(data_dir, name + '_new.wav')
            text = open(os.path.join(data_dir, name + '.txt')).read().strip()
            if os.path.exists(abs_path):
                fs, data = wavfile.read(abs_path)
                meta = {
                    'audio_filepath': abs_path,
                    'duration': data.shape[0]/fs,
                    'text': text.lower()
                }
                json.dump(meta, fp, ensure_ascii=False)
                fp.write('\n')
            else:
                print(f'{abs_path} does not exist')

