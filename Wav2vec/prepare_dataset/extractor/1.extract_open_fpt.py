# import soundfile as sf
from scipy.io import wavfile
import librosa
import json
import tqdm
import os

def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)


data_dir = '/home/duonghai/dataset_voice/1.open_fpt'

transcript = os.path.join(data_dir, 'transcriptAll.txt')
audio_dir = os.path.join(data_dir, 'wav')

if not os.path.exists('config'):
    os.mkdir('config')

print(f'Extracting {data_dir}...')
with open('config/1.open_fpt.config', 'w', encoding='utf8') as fp:
    for line in tqdm.tqdm(open(transcript).read().strip().split('\n')):
        try:
            # print('[+]', line)
            name, text, duration = line.split('|')
            start, end = duration.split('-')
            start = float(start)
            end = float(end)
        except:
            pass

        else:
            if os.path.exists(os.path.join(audio_dir, name[:-4] + '.wav')):
                if not os.path.exists(os.path.join(audio_dir, name[:-4] + '_new.wav')):
                    fs, data = wavfile.read(os.path.join(audio_dir, name[:-4] + '.wav'))
                    data_new = data[int(start*fs) : int(end*fs)]
                    wavfile.write(os.path.join(audio_dir, name[:-4] + '_new.wav'), fs, data_new)
                else:
                    fs, data_new = wavfile.read(os.path.join(audio_dir, name[:-4] + '_new.wav'))

                # print(f'    - name: {name}')
                # print(f'    - text: {text}')
                # print(f'    - start: {start}')
                # print(f'    - end: {end}')
                meta = {
                    'audio_filepath': os.path.join(audio_dir, name[:-4] + '_new.wav'),
                    'duration': data_new.shape[0]/fs,
                    'text': text.lower()
                }
                json.dump(meta, fp, ensure_ascii=False)
                fp.write('\n')

                if data_new.shape[0]/fs < 0.01:
                    print(line)
                    print(meta['audio_filepath'])

                if abs(float(end) - float(start) - data_new.shape[0]/fs) > 0.2:
                    print('mismatch, {} vs {}'.format(float(end) - float(start), data_new.shape[0]/fs))
                # break
        