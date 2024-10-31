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

data_dir = '/media/storage/hai/dataset/8.labeled_youtube'

with open(f'config/youtube_labelled.json', 'w', encoding='utf8') as fp:
    audios = [f for f in recursive_walk(data_dir) if f.endswith('.wav')]
    for f in tqdm.tqdm(audios):
        script_name = f[:-4] + '.json'
        meta = json.loads(open(script_name).read().strip())
        new_meta = {
            'audio_filepath': f,
            'text': meta['transciption'],
            'duration': meta['duration']
        }
        json.dump(new_meta, fp, ensure_ascii=False)
        fp.write('\n')