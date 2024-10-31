import os
import re
import json

def has_numbers(string):
    return bool(re.search(r'\d', string))

def normalize(s):
    text = s.upper().replace('У', 'Y').lower()
    text = text.replace('ν', 'v')
    text = text.replace('ð', 'đ')
    text = text.replace('ϲ', 'c')
    text = text.replace('ρ', 'p')
    text = text.replace('ĺ', '')
     
    return text.lower()

def clean_text(string):
    text = re.sub(r'\_+', ' ', string)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = text.replace('ko', 'không')
    text = text.replace('kô', 'không')
    return normalize(text).strip()

def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)



def read_config(filename):
    with open(filename) as fid:
        return [json.loads(l) for l in fid]

CMD_CONVERT_1CH = 'ffmpeg -y -i "{}" -ac 1 -ar 16000 "{}" > /dev/null 2>&1 < /dev/null'
CMD_CONVERT_NCH = 'ffmpeg -y -i "{}" -ar 16000 "{}" > /dev/null 2>&1 < /dev/null'