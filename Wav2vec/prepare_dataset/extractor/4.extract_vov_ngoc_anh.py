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

data_dir = '/media/storage/hai/dataset/4.tts_VOV_NGOC_ANH'

with open(f'config/vov_ngoc_anh.json', 'w', encoding='utf8') as fp:
    audios = [f for f in recursive_walk(data_dir) if f.endswith('.wav')]
    for f in tqdm.tqdm(audios):
        script_name = os.path.basename(f).replace('audio', 'script').replace('wav', 'json')
        script_name = os.path.join(os.path.dirname(f), script_name)
        meta = json.loads(open(script_name).read().strip())
        new_meta = {
            'audio_filepath': f,
            'text': meta['text'],
            'duration': meta['duration']
        }
        json.dump(new_meta, fp, ensure_ascii=False)
        fp.write('\n')