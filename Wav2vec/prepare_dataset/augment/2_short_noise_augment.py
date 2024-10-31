from audiomentations import AddShortNoises, PolarityInversion
import torchaudio
import csv
import tqdm
import random
import numpy as np
import torch
import os


def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            if 'noise_train' in f or 'noise_val' in f or 'augment_train' in f or 'augment_val' in f:
                continue
            if f.endswith("train.csv") or f.endswith("val.csv"):
                yield os.path.join(r, f)


data_path = '/home/ndanh/STT_dataset'
transform = AddShortNoises(
    sounds_path=[data_path + '/musan/music/all',
                 data_path + '/musan/noise/all',
                 data_path + '/musan/short_noise'],
    min_snr_in_db=3.0,
    max_snr_in_db=20.0,
    noise_rms="relative_to_whole_input",
    min_time_between_sounds=1.5,
    max_time_between_sounds=3.0,
    noise_transform=PolarityInversion(),
    p=1.0
)

# ALERT: should be replace dataset_cmc for get newest csv clean data
augment_csvs = [f for f in recursive_walk('../dataset_cmc')]

output_dir = '/home/ndanh/STT_dataset/52.short_noise'

for mode in ['train', 'val']:
    error = 0
    with open(f'../dataset_cmc/short_noise_13_11_2023_{mode}.csv', 'w') as fp:
        for csv in augment_csvs:
            if csv.endswith(f'{mode}.csv'):
                lines = open(csv).read().strip().split('\n')[1:]
                random.shuffle(lines)
                pbar = tqdm.tqdm(lines)
                pbar.set_description(os.path.basename(csv))
                for idx, line in enumerate(pbar):
                    path, text, duration = line.split(',')
                    if float(duration) < 4:
                        continue

                    pbar.set_postfix(error=error)

                    new_path = f'{output_dir}/{idx}_{os.path.basename(path)[:-4]}_short_noise.wav'

                    data, fs = torchaudio.load(path)
                    assert data.shape[0] == 1 and fs == 16000
                    try:
                        new_data = transform(samples=data[0].numpy().astype(np.float32), sample_rate=fs)
                    except:
                        error += 1
                        continue
                    if isinstance(new_data, np.ndarray):
                        new_data = torch.from_numpy(new_data).unsqueeze(0)
                    torchaudio.save(new_path, new_data, fs, encoding="PCM_S", bits_per_sample=16)
                    print(f'{new_path},{text},{duration}', file=fp)
