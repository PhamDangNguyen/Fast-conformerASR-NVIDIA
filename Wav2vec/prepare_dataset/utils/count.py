import os
import tqdm
import torchaudio


def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)


# for dataset in recursive_walk('../dataset/'):
#     if not 'tiktok' in dataset:
#         continue

#     if not dataset.endswith('_train.csv'):
#         continue

#     print('Counting dataset:', dataset)
#     total = 0.0
#     for line in open(dataset).read().strip().split('\n')[1:]:
#         duration = float(line.split(',')[-1])
#         total += duration
#     print(total, total / 3600)

count = 0
with open('fs48khz.csv', 'w') as fp:
    for wav in recursive_walk('/media/storage/hai/dataset/'):
        if not wav.endswith('.wav'):
            continue
        if 'japanese' in wav:
            continue

        _, sr = torchaudio.load(wav)
        if sr == 48000:
            if count % 10 == 0:
                print(count, wav)
            count += 1
            print(wav, file=fp)
