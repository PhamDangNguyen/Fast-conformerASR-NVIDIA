"""
Augment technique: GaussianNoise, TimeStretch, PitchShift, Shift
check origin in: https://github.com/iver56/audiomentations
Latest update: 8/2023
"""

from audiomentations import *
import numpy as np
import torchaudio
import random
import torch
import tqdm
import os


# augment = Compose([
#     AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
#     TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
#     PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
#     Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
# ])

# # Generate 2 seconds of dummy audio for the sake of example
# samples = np.random.uniform(low=-0.2, high=0.2, size=(32000,)).astype(np.float32)

# # Augment/transform/perturb the audio data
# augmented_samples = augment(samples=samples, sample_rate=16000)

# # torchaudio.save(output_name, noisy, sample_rate, encoding="PCM_S", bits_per_sample=16)

class HaiCompression():
    SUPPORTED_BITRATES = [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        66,
        80,
        96,
        112,
        128,
        144,
        160,
        174,
        192,
        224,
        256,
        320,
    ]

    def __init__(self, codec):
        assert codec in ['aac', 'opus', 'ogg', 'alac']
        self.codec = codec

    def __call__(self, filename, output_name):
        tmp_name = 'tmp'
        bitrate = random.choice(self.SUPPORTED_BITRATES)
        if self.codec == 'aac':
            ext = 'm4a'
            cmd = f'ffmpeg -i "{filename}" -vn -c:a aac -b:a {bitrate}k -map_metadata 0 -ac 1 -ar 16000 -loglevel 0 -y -nostats "{tmp_name}.{ext}" > /dev/null'
        elif self.codec == 'ogg':
            ext = 'ogg'
            cmd = f'ffmpeg -i "{filename}" -vn -c:a libvorbis -q:a {bitrate}k -map_metadata 0 -ac 1 -ar 16000 -loglevel 0 -y -nostats "{tmp_name}.{ext}" > /dev/null'
        elif self.codec == 'opus':
            ext = 'opus'
            cmd = f'ffmpeg -i "{filename}" -vn -c:a libopus -b:a {bitrate}k -map_metadata 0 -ac 1 -ar 16000 -loglevel 0 -y -nostats "{tmp_name}.{ext}" > /dev/null'
        elif self.codec == 'alac':
            ext = 'm4a'
            cmd = f'ffmpeg -i "{filename}" -vn -c:a alac -compression_level {bitrate}k -map_metadata 0 -ac 1 -ar 16000 -loglevel 0 -y -nostats "{tmp_name}.{ext}" > /dev/null'
        cmd_back = f'ffmpeg -y -i "{tmp_name}.{ext}" -ac 1 -ar 16000 "{output_name}" > /dev/null 2>&1 < /dev/null'

        os.system(f'rm -rf "{tmp_name}.{ext}" {output_name}')
        os.system(cmd)
        # assert os.path.exists(f'{tmp_name}.{ext}')
        # print(cmd)
        os.system(cmd_back)
        assert os.path.exists(output_name)
        # print(cmd_back)
        # exit()


AUGMENT_FUNC = {
    'time_stretch': TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0),
    'time_mask': TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=1.0),
    'mp3_compression': Mp3Compression(min_bitrate=8, max_bitrate=320, p=1.0),
    'hai_compresison_aac': HaiCompression('aac'),
    'hai_compresison_alac': HaiCompression('alac'),
}

METHODS = list(AUGMENT_FUNC.keys())

augment_csvs = [
    # '../dataset/mix_train.csv',
    '../dataset_cmc/nghiem_thu_ytb_2022_02_train.csv',
    '../dataset_cmc/verified_cmc_prediction_tiktok_crawl_orig_1ch_train.csv',
    '../dataset_cmc/youtube_labelled_train.csv',
    '../dataset_cmc/wo_space_path_tts_pham_nguyen_son_tung_train.csv',
    '../dataset_cmc/verified_fleur_train.csv',
    '../dataset_cmc/youtube_12_2022_train.csv',
]

output_dir = '/home/ndanh/STT_dataset/35.augment_08_2023'


def augment():
    error = 0
    with open('new_augment_08_2023.csv', 'w') as fp:
        fp.write('file,text,duration\n')
        for csv in augment_csvs:
            lines = open(csv).read().strip().split('\n')[1:]
            random.shuffle(lines)
            pbar = tqdm.tqdm(lines)
            pbar.set_description(os.path.basename(csv))
            for idx, line in enumerate(pbar):
                path, text, duration = line.split(',')
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


def split_train_val():
    bar = open('new_augment_08_2023.csv').read().strip().split('\n')[1:]
    random.shuffle(bar)
    split = int(0.2 * len(bar))
    dev_split = bar[:split]
    train_split = bar[split:]
    with open('/home/ndanh/asr-wav2vec/dataset_cmc/augment_compression_08_2023_train.csv', 'w') as f1, open(
            '/home/ndanh/asr-wav2vec/dataset_cmc/augment_compression_08_2023_val.csv', 'w') as f2:
        f1.write('file,text,duration\n')
        f2.write('file,text,duration\n')
        for line in train_split:
            path, label, duration = line.split(',')
            f1.write(f'{path},{label},{duration}\n')

        for line in dev_split:
            path, label, duration = line.split(',')
            f2.write(f'{path},{label},{duration}\n')


if __name__ == '__main__':
    augment()
    split_train_val()
    os.remove('new_augment_08_2023.csv')
