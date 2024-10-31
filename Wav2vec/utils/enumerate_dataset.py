"""
Display all information of dataset
"""

import os
import tqdm
import soundfile as sf


def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)


with open('stt_dataset_info.csv', 'w') as fp:
    fp.write('idx,path,samples,duration (hours)\n')
    count = 1
    for dataset in recursive_walk('/home/dvhai/code/wav2vec2/dataset'):
        if dataset.endswith('_train.csv'):
            name = os.path.basename(dataset.replace('_train.csv', ''))
            datasets = [dataset, dataset.replace('_train.csv', '_val.csv')]
            print(count, name, datasets)

            duration = 0.0
            samples = 0
            for ds in datasets:
                if not os.path.exists(ds):
                    continue
                ls = tqdm.tqdm(open(ds).read().strip().split('\n')[1:])
                for line in ls:
                    path, text, d = line.split(',')
                    assert os.path.exists(path), print(path)
                    speech, fs = sf.read(path)
                    d = len(speech) / fs
                    if float(d) > 20:
                        print(ds)
                        print(path)
                        print()
                    duration += float(d)

                samples += len(ls)
            fp.write(f'{count},{name},{samples},{duration / 3600:.02f}\n')

            count += 1
