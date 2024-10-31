from joblib import Parallel, delayed
from scipy.io import wavfile
import librosa
import json
import tqdm
import os

import argparse

INDEX = {
    'vtc': 10, 
    'VTC1': 11, 
    'vtv_24': 12, 
    'VTV4': 13, 
}

def get_args():
    # create argument parser
    parser = argparse.ArgumentParser()

    # parameter for dataset
    parser.add_argument('--dataset', type=str, default='vtc',
                        choices=['vtc', 'VTC1', 'vtv_24', 'VTV4'])

    # parameter for workers
    parser.add_argument('--num_worker', type=int, default=64)

    # parse args
    args = parser.parse_args()

    # add path
    args.index       = str(INDEX[args.dataset])
    args.data_dir    = f'/home/duonghai/{args.dataset}'
    args.output_dir  = f'/media/dataDrive/haidv/dataset_voice/{args.index}.{args.dataset}'
    args.config_file = f'/home/duonghai/code/extractor/config/{args.index}.{args.dataset}.config'
    args.tmp_folder  = f'/home/duonghai/code/extractor/tmp/{args.index}.{args.dataset}'

    return args

def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)

def read_json(path):
    return json.loads(open(path).read())

def worker(path_id, path, num_path, args):
    try:
        # extract worker id
        worker_id = path_id % args.num_worker

        # display some info
        print(f'[+] progress={path_id}/{num_path} worker={worker_id} path={path}')

        # convert mp3 to wav
        mp3_path = path.replace('json', 'mp3').replace('scripts', 'audio')
        wav_path = mp3_path.replace('mp3', 'wav').replace(args.data_dir, args.output_dir)
        # make output folder
        wav_folder = os.path.dirname(wav_path)
            
        if not os.path.exists(wav_folder):
            os.makedirs(wav_folder)
            cmd = f'ffmpeg -i "{mp3_path}" -ac 1 -ar 16000 "{wav_path}" > /dev/null 2>&1'
            if not os.path.exists(wav_path):
                os.system(cmd)

            # doc file json
            meta = read_json(path)[0]
            for items in meta.values():
                pass

            # doc file wav
            sr, data = wavfile.read(wav_path)
            for i, item in enumerate(items):
                # write wav
                segment = data[int(item['start'] * sr): int((item['start'] + item['duration']) * sr)]
                segment_path = wav_path.replace('.wav', f'_{i}.wav')
                wavfile.write(segment_path, sr, segment)
                # write config
                tmp_path = os.path.join(args.tmp_folder, f'{worker_id}.config')
                with open(tmp_path, 'a+', encoding='utf8') as fp:
                    segment_meta = {
                        'audio_filepath': segment_path,
                        'duration': item['duration'],
                        'text': item['text'].lower()
                    }
                    json.dump(segment_meta, fp, ensure_ascii=False)
                    fp.write('\n')

            # remove file wav
            if os.path.exists(wav_path):
                os.remove(wav_path)
    except:
        pass

def main():
    # get args
    args = get_args()

    # input
    transcript_paths = [j for j in recursive_walk(args.data_dir) if j.endswith('.json')]    
    num_path = len(transcript_paths)

    # make output folder
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.tmp_folder):
        os.makedirs(args.tmp_folder)

    # parallel
    Parallel(n_jobs=args.num_worker)(delayed(worker)(path_id, path, num_path, args) for path_id, path in enumerate(transcript_paths))

if __name__ == '__main__':
    main()
