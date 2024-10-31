import tqdm
from utils.wer import wer
from scipy.io import wavfile

count_valid = 0
coutn_invalid = 0
bar = tqdm.tqdm(open('common_voice_infer_False.csv').read().strip().split('\n')[1:])
with open('test/common_voice.csv', 'w') as f1, open('test/common_voice_invalid.csv', 'w') as f2:
    f1.write('file,text,duration\n')
    f2.write('file,text,duration\n')
    for line in bar:
        bar.set_postfix(valid=count_valid, invalid=coutn_invalid)
        path, label, predict = line.split(',')
        # print(path, label, predict)
        try:
            w = wer([label], [predict], use_tqdm=False)
        except:
            coutn_invalid += 1
            continue
        # print(w)
        fs, speech = wavfile.read(path)
        duration = len(speech) / fs
        if w < 25:
            count_valid += 1
            f1.write(f'{path},{label},{duration}\n')
        else:
            coutn_invalid += 1
            f2.write(f'{path},{label},{duration}\n')

    # break