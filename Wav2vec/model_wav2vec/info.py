import os

def get_info(dataset_folder, postfix='train'):
    count_audio = 0
    total_duration = 0.0
    min_duration = 99999
    max_duration = 0
    for file in os.listdir(dataset_folder):
        if not postfix in file:
            continue

        if 'youtube_en' in file:
            continue
        
        print('[+] Processing:', file)

        filepath = os.path.join(dataset_folder, file)
        for line in open(filepath).read().strip().split('\n'):
            audio, text, duration = line.split(',')
            if not audio.endswith('.wav'):
                continue
            count_audio += 1
            duration = float(duration)
            total_duration += duration
            
            if duration < min_duration:
                min_duration = duration
            
            if duration > max_duration:
                max_duration = duration

    with open(f'info_{postfix}.txt', 'w') as fp:
        fp.write(f'Utterance: {count_audio}\n')
        fp.write(f'Duration: {total_duration/3600:.02f} (hours)\n')
        fp.write(f'Max duration: {max_duration:.02f} (seconds)\n')
        fp.write(f'Min duration: {min_duration:.02f} (seconds)\n')


if __name__ == '__main__':
    dataset_folder = '/home/dvhai/code/wav2vec2/dataset'
    get_info(dataset_folder, 'train')
    get_info(dataset_folder, 'val')
    