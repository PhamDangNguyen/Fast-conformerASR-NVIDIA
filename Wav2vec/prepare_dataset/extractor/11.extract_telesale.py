import unicodedata as ud
import json
import tqdm
import os
import re

def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)

data_dir = '/media/storage/hai/dataset/10.telesale'

with open(f'/home/dvhai/code/nemo_asr/config/dataset/telesale_fix_nfc.json', 'w', encoding='utf8') as fp:
    audios = [f for f in recursive_walk(data_dir) if f.endswith('.wav')]
    for f in tqdm.tqdm(audios):
        script_name = os.path.basename(f).replace('wav', 'json')
        script_name = os.path.join(os.path.dirname(f), script_name)
        meta = json.loads(open(script_name).read().strip())
        if len(meta['transciption']) > 15:
            new_meta = {
                'audio_filepath': f,
                'text': re.sub(r' +', ' ', re.sub(r'[^\w\s]', ' ', ud.normalize('NFC', meta['transciption'].replace('-', ' ')).strip())),
                'duration': meta['duration']
            }
            json.dump(new_meta, fp, ensure_ascii=False)
            fp.write('\n')
