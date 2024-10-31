import numpy as np
import os
import random

def padding(speech, duration=10):
    if len(speech) < duration * 16000:
        return np.concatenate((speech, np.zeros(duration*16000 - len(speech))))
    if len(speech) > duration * 16000:
        return speech[:duration*16000]
    return speech


def get_random_example():
    # return '/media/storage/hai/dataset/fpt/nam/nu/140290_linhsan_nu_nam_new_bg_noise_denoise_dns48.wav,vì vậy dư luận phỏng đoán rất có thể khương chu cũng liên đới đến nhóm lợi ích của bạc hy lai,5.0650625'.split(',')
    dataset_dir = '../dataset'
    csvs = [f for f in os.listdir(dataset_dir) if 'val' in f]
    csv = random.choice(csvs)
    files = open(os.path.join(dataset_dir, csv)).read().strip().split('\n')[1:]
    while True:
        file = random.choice(files)
        path, text, duration = file.split(',')
        if 10 < float(duration) < 30:
            break
    # print(file)
    return file.split(',')
