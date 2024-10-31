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

data_dir = r'C:\Users\DatalakeStation\Documents\Projects\youtube-transcript-crawler\cmc_asr\media_cmc'

with open(f'C:\\Users\\DatalakeStation\\Documents\\Projects\\youtube-transcript-crawler\\cmc_asr\\media_cmc.json', 'w', encoding='utf8') as fp:
    audios = [f for f in recursive_walk(data_dir) if f.endswith('_16khz.wav')]
    for f in tqdm.tqdm(audios):
        script_name = os.path.join(os.path.dirname(f), 'transcript.txt')
        fs, data = wavfile.read(f)
        new_meta = {
            'audio_filepath': f,
            'text': open(script_name, encoding='utf-8').read().strip(),
            'duration': float(len(data)/fs)
        }
        json.dump(new_meta, fp, ensure_ascii=False)
        fp.write('\n')