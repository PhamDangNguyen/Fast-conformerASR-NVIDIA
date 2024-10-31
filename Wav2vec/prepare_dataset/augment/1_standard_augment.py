"""
Multiple technique augment audio applied in here
"""
import torch
import torchaudio
import torchaudio.functional as F
import matplotlib.pyplot as plt
from torchaudio.utils import download_asset
import os
import json
import random
import tqdm
from joblib import Parallel, delayed

print(torch.__version__)
print(torchaudio.__version__)



def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)


def read_config(filename):
    with open(filename) as fid:
        return [json.loads(l) for l in fid]


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c + 1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    # plt.show(block=False)
    plt.savefig(f'{title}.png')


"""
Background noise: Hard noise
"""
ADD_BACKGROUND_NOISE = False


def add_noise(speech, noise, sample_rate=16000, snr_db=5):
    assert speech.shape[0] == 1
    assert noise.shape[0] == 1
    assert sample_rate == 16000

    if not ADD_BACKGROUND_NOISE:
        return speech

    while noise.shape[1] < speech.shape[1]:
        noise = torch.concat((noise, noise), axis=1)
    noise = noise[:, : speech.shape[1]]
    scale = (10 ** (snr_db / 20)) * noise.norm(p=2) / speech.norm(p=2)
    noisy = (scale * speech + noise) / 2
    return noisy


def gen_config():
    """
    Create config file of RIR effect + Background noise and split train file
    Returns:

    """
    SNRs = [5, 10, 15]
    DATASET_PERCENT = {
        'train': 1 / 100,
        'val': 1 / 100
    }
    DATASET_CSV_DIR = '../dataset/'
    MAX_SAMPLE = 500

    for TYPE in ['train', 'val']:
        total = 0
        augment_total = 0
        for dataset in [f for f in recursive_walk(DATASET_CSV_DIR) if
                        (f.endswith(f'_{TYPE}.csv') and (not 'noise' in f) and (not 'music' in f))]:
            print('Process:', dataset)
            samples = open(dataset).read().strip().split('\n')[1:]
            total += len(samples)

            for noise_type in ['music', 'noise']:
                for rir_type in ['real_rirs', 'largeroom', 'mediumroom', 'smallroom']:
                    for snr in SNRs:
                        random.shuffle(samples)
                        if TYPE == 'train' and len(samples) > 1000:
                            samples_ = samples[:int(len(samples) * DATASET_PERCENT[TYPE])]
                        elif TYPE == 'val':
                            samples_ = samples[:int(len(samples) * DATASET_PERCENT[TYPE])]

                        samples_ = samples_[:MAX_SAMPLE]

                        noise_files = [f for f in recursive_walk('noise') if (f.endswith('.wav') and noise_type in f)]
                        rir_files = [f for f in recursive_walk('rir') if (f.endswith('.wav') and rir_type in f)]
                        os.makedirs('config', exist_ok=True)
                        print('\t[+] Creating:', f'augment_{noise_type}_{rir_type}_{snr}_{os.path.basename(dataset)}',
                              len(samples_))
                        with open(f'config/augment_{noise_type}_{rir_type}_{snr}_{os.path.basename(dataset)}',
                                  'w') as fp:
                            fp.write('clean,rir,noise,text,snr\n')
                            for sid, sample in enumerate(samples_):
                                clean, text, duration = sample.split(',')
                                if float(duration) < 2:
                                    print('\t\t- Exclude:', clean, 'duration:', duration)
                                    continue
                                noise = random.choice(noise_files)
                                rir = random.choice(rir_files)
                                fp.write(f'{clean},{rir},{noise},{text},{snr}\n')

                        # print(len(samples))
                        augment_total += len(samples_)

        print(TYPE, augment_total, total, augment_total / total)


"""
Add RIR
"""


def add_rir(speech, rir, sample_rate=16000):
    assert speech.shape[0] == 1
    assert rir.shape[0] == 1
    assert sample_rate == 16000

    RIR = rir / torch.norm(rir, p=2)
    RIR = torch.flip(RIR, [1])
    speech_ = torch.nn.functional.pad(speech, (RIR.shape[1] - 1, 0))
    rir_applied = torch.nn.functional.conv1d(speech_[None, ...], RIR[None, ...])[0]
    return rir_applied


def process_one(idx, line, config, output_folder):
    clean_path, rir_path, noise_path, text, snr = line.split(',')
    assert os.path.exists(clean_path)
    assert os.path.exists(rir_path)
    assert os.path.exists(noise_path)

    clean, sample_rate = torchaudio.load(clean_path)
    assert sample_rate == 16000

    rir, sample_rate = torchaudio.load(rir_path)
    assert sample_rate == 16000

    noise, sample_rate = torchaudio.load(noise_path)
    assert sample_rate == 16000

    try:
        if ADD_BACKGROUND_NOISE:
            rir_applied = add_rir(clean, rir, sample_rate)
            noisy = add_noise(rir_applied, noise, sample_rate, int(snr))
            output_name = f'{output_folder}/{os.path.basename(config)[:-4]}_{idx}.wav'
            torchaudio.save(output_name, noisy, sample_rate, encoding="PCM_S", bits_per_sample=16)
            duration = noisy.shape[1] / sample_rate
            if idx > 0 and idx % 10 == 0:
                print('\t-', output_name)

            return f'{output_name},{text},{duration}\n'

        if idx % 5 == 0:
            rir_applied = add_rir(clean, rir, sample_rate)
            noisy = add_noise(rir_applied, noise, sample_rate, int(snr))
            output_name = f'{output_folder}/{os.path.basename(config)[:-4]}_wo_bg_{idx}.wav'
            torchaudio.save(output_name, noisy, sample_rate, encoding="PCM_S", bits_per_sample=16)
            duration = noisy.shape[1] / sample_rate
            print('\t-', output_name)

            return f'{output_name},{text},{duration}\n'

    except RuntimeError:
        return ''

    return ''


def augment(config, fp, output_folder='output'):
    os.makedirs(output_folder, exist_ok=True)
    # pbar = tqdm.tqdm(open(config).read().strip().split('\n')[1:])
    lines = open(config).read().strip().split('\n')[1:]

    # res = 
    with Parallel(n_jobs=os.cpu_count()) as parallel:
        results = parallel(delayed(process_one)(idx, line, config, output_folder) for idx, line in enumerate(lines))

    count = 0
    for res in results:
        if res != '':
            count += 1
            fp.write(res)

    print('\t[+] Done:', config, 'count:', count)


def split_train_val(csv_file):
    lines = open(csv_file).read().strip().split('\n')[1:]
    for dataset_type in ['train', 'val']:
        pattern = f'_{dataset_type}_'
        output_csv = f'{csv_file[:-4]}_{dataset_type}.csv'
        print(f'[+] Processing {csv_file} ==> {output_csv}')
        pbar = tqdm.tqdm([l for l in lines if (pattern in l)])

        with open(output_csv, 'w') as fp:
            fp.write('file,text,duration\n')
            for line in pbar:
                assert os.path.exists(line.split(',')[0])
                fp.write(f'{line}\n')

    # gen_config()


def generate_rir_data():
    if ADD_BACKGROUND_NOISE:
        OUTPUT_FOLDER = '/media/storage/hai/dataset/26.augment'
        OUTPUT_CSV = 'augment.csv'
    else:
        OUTPUT_FOLDER = '/media/storage/hai/dataset/27.augment_wo_bg'
        OUTPUT_CSV = 'augment_wo_bg.csv'

    configs = list(recursive_walk('config/'))
    total_configs = len(configs)
    with open(OUTPUT_CSV, 'w') as fp:
        fp.write('file,text,duration\n')
        for idx, config in enumerate(configs):
            print(f'[{idx}/{total_configs}] Processing {config}')
            augment(config, fp, output_folder=OUTPUT_FOLDER)

    split_train_val(OUTPUT_CSV)


"""
CODEC HERE
"""


def apply_codec(noisy, sample_rate, config):
    effects = [
        ["lowpass", "4000"],
        [
            "compand",
            "0.02,0.05",
            "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8",
            "-8",
            "-7",
            "0.05",
        ],
        ["rate", "8000"],
    ]

    downsampled = torchaudio.transforms.Resample(sample_rate, 8000)(noisy)

    filtered, _ = torchaudio.sox_effects.apply_effects_tensor(
        downsampled,
        8000,
        effects=effects,
    )

    codec_applied = F.apply_codec(filtered, 8000, **config)

    upsampled = torchaudio.transforms.Resample(8000, sample_rate)(codec_applied)

    return upsampled


def generate_codec(path, fp=None, shuffle=False):
    OUTPUT_FOLDER = '/media/storage/hai/dataset/28.augment_codec'
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    lines = open(path).read().strip().split('\n')[1:]
    total_samples = len(lines)
    if shuffle:
        random.shuffle(lines)
    lines = lines[:total_samples // 2]

    configs = {
        'gsm': {"format": "gsm"},
        'vorbis': {"format": "vorbis", "compression": -1},
        'ulaw': {"format": "wav", "encoding": "ULAW", "bits_per_sample": 8},
    }

    pbar = tqdm.tqdm(lines)
    for line in pbar:
        file, text, duration = line.split(',')
        waveform, sample_rate = torchaudio.load(file)
        assert sample_rate == 16000

        for codec_type in configs:
            output_name = f'{os.path.join(OUTPUT_FOLDER, os.path.basename(file))[:-4]}_codec_{codec_type}.wav'
            codec_applied = apply_codec(waveform, sample_rate, configs[codec_type])
            torchaudio.save(output_name, codec_applied, sample_rate, encoding="PCM_S", bits_per_sample=16)
            fp.write(f'{output_name},{text},{duration}\n')


if __name__ == '__main__':
    OUTPUT_CSV = 'augment_codec_wo_bg.csv'
    print('[+] Generating')
    with open(OUTPUT_CSV, 'w') as fp:
        fp.write('file,text,duration\n')
        # generate_codec('augment.csv', fp=fp, shuffle=True)
        generate_codec('augment_wo_bg.csv', fp=fp, shuffle=True)

    # split_train_val(OUTPUT_CSV)
    print('[+] Splitting train/val')
    lines = open(OUTPUT_CSV).read().strip().split('\n')[1:]
    random.shuffle(lines)
    # for t in ['val', 'train']:
    train_samples = lines[:-500]
    val_samples = lines[-500:]

    with open('augment_codec_wo_bg_train.csv', 'w') as fp:
        fp.write('file,text,duration\n')
        for line in tqdm.tqdm(train_samples):
            fp.write(f'{line}\n')

    with open('augment_codec_wo_bg_val.csv', 'w') as fp:
        fp.write('file,text,duration\n')
        for line in tqdm.tqdm(val_samples):
            fp.write(f'{line}\n')

# Origin from torch audio
# SAMPLE_RIR      = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")
# SAMPLE_SPEECH   = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042-8000hz.wav")
# SAMPLE_NOISE    = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")

# SAMPLE_RIR      = '/home/dvhai/code/wav2vec2/augment/rir/real_rirs/RWCP_type2_rir_cirline_jr1_imp090-13.wav'
# SAMPLE_SPEECH   = '/media/storage/hai/dataset/23.youtube_12_2022/datasets_trithunhanloai/PLnRl-W3gZI79kfp8E7lcDkImtMHA6FIfr/GcbMPtEq8RE/wav/GcbMPtEq8RE_63.wav'
# SAMPLE_NOISE    = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")


# speech, sample_rate = torchaudio.load(SAMPLE_SPEECH)
# print(speech.shape, sample_rate)

# noise, _ = torchaudio.load(SAMPLE_NOISE)
# print(noise.shape, sample_rate)

# add_noise(speech, noise, sample_rate=sample_rate)
# rir, sample_rate = torchaudio.load(SAMPLE_RIR)
# RIR = rir / torch.norm(rir, p=2)
# RIR = torch.flip(RIR, [1])
# print(RIR.shape, sample_rate)


# speech_ = torch.nn.functional.pad(speech, (RIR.shape[1] - 1, 0))
# augmented = torch.nn.functional.conv1d(speech_[None, ...], RIR[None, ...])[0]
# print(augmented.shape)

# plot_waveform(RIR, sample_rate, title='rir')
# plot_waveform(speech, sample_rate, title='raw')
# plot_waveform(augmented, sample_rate, title='augmented')

# torchaudio.save('speech.wav', speech, sample_rate, encoding="PCM_S", bits_per_sample=16)
# torchaudio.save('augmented.wav', augmented, sample_rate, encoding="PCM_S", bits_per_sample=16)
