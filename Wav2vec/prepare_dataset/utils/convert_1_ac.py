"""
Convert 1 channel and sample rate 16000hz
"""
import glob
import os
from scipy.io import wavfile


def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)


def convert_16k_1ac():
    audio_paths = glob.glob(f"/home/ndanh/asr-wav2vec/File_bảo_mật/*")
    for audio_path in audio_paths:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        save_name = f'/home/ndanh/asr-wav2vec/File_bảo_mật/{base_name}.wav'
        command = f'ffmpeg -i {audio_path} -ar 16000 -ac 1 -resampler soxr {save_name}'
        os.system(command)


CMD_CONVERT_1CH = 'ffmpeg -y -i "{}" -ac 1 -ar 16000 "{}" > /dev/null 2>&1 < /dev/null'


def generate(csv_name, old_root, new_root):
    print('Converting:', csv_name)
    with open(f'{csv_name[:-4]}_1ch.csv', 'w', encoding='utf8') as fp:
        print(f'file,text,duration', file=fp)
        pbar = tqdm.tqdm(open(csv_name).read().strip().split('\n')[1:])
        for line in pbar:
            path, text, _ = line.split(',')
            if len(text) < 5:
                continue
            path = path.replace(old_root, new_root)
            assert os.path.exists(path)

            new_path = path[:-4] + '_1ch.wav'
            if not os.path.exists(new_path):
                os.system(CMD_CONVERT_1CH.format(path, new_path))

            assert os.path.exists(new_path)

            fs, speech = wavfile.read(new_path)
            duration = len(speech) / fs

            if duration > 20 or duration < 1:
                continue

            print(f'{new_path},{text},{duration}', file=fp)


if __name__ == '__main__':
    # convert_16k_1ac()
    audio_path = "/home/ndanh/asr-wav2vec/CX_02_031.mp3"
    save_name = "/home/ndanh/asr-wav2vec/CX_02_031.wav"
    command = f'ffmpeg -i {audio_path} -ar 16000 -ac 1 {save_name}'
    os.system(command)

    # csv_files = [f for f in recursive_walk('dataset_cmc') \
    #                        if f.endswith(f'_train.csv')]
    # print(csv_files)

    # data = pd.concat([pd.read_csv(f) for f in csv_files], axis=0, ignore_index=True)

    # for idx in range(len(data)):
    #     batch = data.iloc[idx].copy()
    #     batch = batch.to_dict()
    #     speech_array, sampling_rate = sf.read(batch['file'])
    #     if speech_array is None:
    #         print(batch)
    #         sys.exit()

    # audio, sr = sf.read("/home/ndanh/STT_dataset/52.short_noise/36965_3693554_minhquang_nam_nam_new_bg_noise_denoise_dns48_short_noise.wav")
    # print(audio.shape)

    # csv_files = [f for f in recursive_walk('dataset_cmc_remove_duplicate') \
    #                        if f.endswith(f'_train.csv')]
    # data = pd.concat([pd.read_csv(f) for f in csv_files], axis=0, ignore_index=True)

    # data = data[1 < data['duration']].reset_index(drop=True)
    # data = data[data['duration'] <= 20].reset_index(drop=True)

    # data.to_csv('all.csv')


    # ========================
    # csv_name = 'final_tiktok_label_filter_3.csv'
    # # old_root = '/home/ndanh/downloaded_video_2/'
    # old_root = 'downloaded_video_3/'
    # new_root = '/media/storage/hai/dataset/32.tiktok/downloaded_video_3/'
    # generate(csv_name, old_root, new_root)
