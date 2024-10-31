from typing import Any
from audiomentations import *
import numpy as np
import torchaudio
import random
import torch
import tqdm
import os


def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            if "augment_codec" in f or "duyanh_new_augment_08_2023" in f or 'noise_train' in f or 'noise_val' in f:
                continue
            if f.endswith("train.csv") or f.endswith("val.csv"):
                yield os.path.join(r, f)


class DuyAnhCompression:
    def __init__(self):
        print("Using speed changing in torchaudio")

    def __call__(self, filename, output_name):
        waveform, sample_rate = torchaudio.load(filename)
        speed_change = random.uniform(0.75, 1.4)
        speed_change = f"{speed_change}"
        if random.random() > 0.8:

            effects = [
                ["speed", speed_change],  # increase the speed
                # This only changes sample rate, so it is necessary to
                # add `rate` effect with original sample rate after this.
                ["rate", f"{sample_rate}"],
                ["reverb", "-w"],  # Reverbration gives some dramatic feeling
            ]
        else:
            effects = [
                ["speed", speed_change],  # increase the speed
                # This only changes sample rate, so it is necessary to
                # add `rate` effect with original sample rate after this.
                ["rate", f"{sample_rate}"],
                # ["reverb", "-w"],  # Reverbration gives some dramatic feeling
            ]
        waveform2, sample_rate2 = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects)
        torchaudio.save(output_name, waveform2, sample_rate2, encoding="PCM_S", bits_per_sample=16)


augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=1.0)
])

AUGMENT_FUNC = {
    'audioaugmentation': augment,
    'duyanh_augment': DuyAnhCompression()
}

METHODS = list(AUGMENT_FUNC.keys())

augment_csvs = [f for f in recursive_walk('../dataset_cmc')]
print(augment_csvs)

output_dir = '/home/ndanh/STT_dataset/56.augment_13_11_2023'
for mode in ['train', 'val']:
    error = 0
    with open(f'../dataset_cmc/new_augment_13_11_2023_{mode}.csv', 'w') as fp:
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
                    method = METHODS[idx % len(METHODS)]
                    func = AUGMENT_FUNC[method]
                    pbar.set_postfix(method=method, error=error)
                    # print(method, func)
                    # print(path)
                    new_path = f'{output_dir}/{idx}_{os.path.basename(path)[:-4]}_{method}.wav'
                    # print(new_path)
                    # exit()
                    if method.startswith('hai_'):
                        func(path, new_path)
                        if not os.path.exists(new_path):
                            error += 1
                            continue
                    elif method.startswith('duyanh_'):
                        func(path, new_path)
                        if not os.path.exists(new_path):
                            error += 1
                            continue
                    else:
                        data, fs = torchaudio.load(path)
                        assert data.shape[0] == 1 and fs == 16000
                        try:
                            new_data = func(samples=data.numpy().astype(np.float32), sample_rate=fs)
                        except:
                            error += 1
                            continue
                        if isinstance(new_data, np.ndarray):
                            new_data = torch.from_numpy(new_data)
                        torchaudio.save(new_path, new_data, fs, encoding="PCM_S", bits_per_sample=16)
                    print(f'{new_path},{text},{duration}', file=fp)
