from joblib import Parallel, delayed
from math import floor, ceil
from scipy.io import wavfile
import soundfile as sf
import traceback
import webvtt
import json
import tqdm
import sys
import os
import re


def convert_vtt_time_to_second(time_vtt, minus_time=False):
    hour, minutes, second = time_vtt.split(":")
    minus_total_time = 0.2 if minus_time else 0
    return max(float(hour)*3600 + float(minutes)*60 + float(second) - minus_total_time, 0.0)


def normalize_automatic_subtitle_google(text):
    text = text.lower().strip()
    remove_accent = ['ừ', 'ờ', 'à', 'em', 'a', 'ạ', 'á', 'ê', 'mẹ', 'bộ', 'nghe', 'ở', 'ý', 'ô']
    last_word = text.split(' ')[-1]
    if last_word in remove_accent:
        text = text.replace(last_word, '', 1).strip()
    while True:
        first_word = text.split(' ')[0]
        if first_word in remove_accent:
            text = text.replace(first_word, '', 1).strip()
        else:
            break

    heuristic_norm_text = [('có viết 19', 'covid 19'), ('cô viết', 'covid'), ('cô biết', 'covid'), ('coffee 2', 'sars-cov 2'),
                               ('office 19', 'covid 19'), ('sắc cô vi', 'sars-cov 2'), ('19a', '19'),
                               ('không biết 19', 'covid 19'), ('cô vẫn 19', 'covid'), ('sắc cô', 'sars-cov'),
                               ('có biết 19', 'covid 19'), ('cô vít', 'covid'), ('covich', 'covid'), ("[Nhạc]", ''), ('gì có vi', 'dịch covid'),
                           ('coban 19', 'covid 19')]
    for x in heuristic_norm_text:
        text = text.replace(x[0], x[1])
    return text



def read_line_vtt(file_path, meta_path):
    with open(file_path, 'r', encoding='utf-8') as fp:
        results = []
        flag_skip = False
        index_skip = 0
        lines = fp.readlines()


        back_string_end = 0
        temp_string = ''
        temp_start = 0

        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
        else:
            meta_data = {'subtitles': []}

        for idx, line in enumerate(lines):
            # print("Line {}: {}".format(idx, line.strip()))
            # print(temp_start)
            if 'subtitles' in meta_data and 'vi' in meta_data['subtitles']:
                # Subtitle tự tạo => Thường được tác giả gán không chuẩn => Không sử dụng
                num_index_skip = 2
                try:
                    if flag_skip:
                        index_skip += 1
                        if index_skip == num_index_skip:
                            flag_skip = False
                    else:
                        match_time = re.match(r'.*?([0-9]{2}:[0-9]{2}:.*?)\s-->\s(.*?)$', line.strip())
                        match_next_time = re.match(r'.*?([0-9]{2}:[0-9]{2}:.*?)\s-->\s(.*?)$', lines[idx+3].strip())
                        if match_time is not None:
                            results.append((convert_vtt_time_to_second(match_time.group(1), minus_time=True),
                                            convert_vtt_time_to_second(match_next_time.group(1), minus_time=True),
                                            lines[idx+1].strip()))
                            flag_skip = True
                            index_skip = 0

                except Exception as e:
                    traceback.print_exc()
                    continue
            else:
                # Auto generate subtitle
                num_index_skip = 6
                try:
                    if flag_skip:
                        index_skip += 1
                        if index_skip == num_index_skip:
                            flag_skip = False
                    else:
                        match_time = re.match(r'.*?([0-9]{2}:[0-9]{2}:.*?)-->\s(.*?)\s.*$', line.strip())
                        # print(match_time)
                        if match_time is not None:

                            # Margin time lấy từ dòng có dạng: 00:00:00.000 --> 00:00:04.390 align:start position:0%
                            margin_start = convert_vtt_time_to_second(match_time.group(1), minus_time=True)
                            margin_end = convert_vtt_time_to_second(match_time.group(2))
                            # print(margin_start, re.findall(r'<([0-9].*?)>', lines[idx + 2]))

                            # Current time lấy từ dòng có dạng: a<00:00:01.310><c> chất</c><00:00:02.310><c>
                            match_real_current_time = re.findall(r'<([0-9].*?)>', lines[idx + 2])
                            if len(match_real_current_time) > 0:
                                current_start = convert_vtt_time_to_second(match_real_current_time[0], minus_time=True)
                                current_end = convert_vtt_time_to_second(match_real_current_time[-1])

                                current_text = normalize_automatic_subtitle_google(' '.join(lines[idx + 5].split()))
                                # print("Current line data: ", current_start, current_end, back_string_end, current_text, idx,
                                #       len(lines))

                                # Lấy text từ dòng idx + 5, có trường hợp data gốc ko có đoạn text đấy => cần check match ở dòng số idx + 2 để lấy bù vào
                                if current_text == '':
                                    text_from_time = re.sub(r'(</c>)?<([0-9].*?)><c>', '', lines[idx + 2]).replace('</c>',
                                                                                                                   '')
                                    if text_from_time.find('[') > 0: continue
                                    current_text = normalize_automatic_subtitle_google(' '.join(text_from_time.split()[1:]))

                                # print(temp_start, current_start, margin_start)
                                if margin_start == 0.0 or current_start - margin_start > 3:
                                    # Câu đầu tiên mà có nhạc quá dài, thỉnh thoảng bị trùng lặp, do từ đầu tiên xuất hiện quá sớm 00:00:00.06
                                    temp_start = current_start
                                    temp_string = current_text
                                    # print("!2121", temp_string, margin_start)

                                if back_string_end != 0 and (current_start - back_string_end > 0.3 or current_start - temp_start > 15 or idx + 6 > len(lines)):
                                    # if back_string_end != 0 and (current_start - back_string_end > 0.3 or idx + 6 > len(lines)):
                                    # ngắt câu tại đây

                                    if all(not c.isalnum() for c in temp_string):
                                        # print("None text, continue!")
                                        temp_start = current_start
                                        # continue
                                    else:
                                        # print("Append full merge string: ", temp_start, back_string_end, repr(temp_string))
                                        results.append((temp_start, back_string_end, temp_string))
                                        # print("temp string: ", temp_string)
                                        temp_start = (current_start + margin_start * 3) / 4 if margin_start - current_start > 0.2 else margin_start

                                        # Kiểm tra điều kiện để bỏ token đầu tiên của 1 câu dài
                                        if len(match_real_current_time) > 2:
                                            compare_first_2_tokens = (convert_vtt_time_to_second(
                                                match_real_current_time[1]) - convert_vtt_time_to_second(
                                                match_real_current_time[0])) > \
                                                                     (convert_vtt_time_to_second(match_real_current_time[
                                                                                                     2]) - convert_vtt_time_to_second(
                                                                         match_real_current_time[1]))

                                            current_text = normalize_automatic_subtitle_google(
                                                ' '.join(lines[idx + 5].split()[1 if compare_first_2_tokens else 0:]))

                                    # setup thời gian câu mới:
                                    temp_string = current_text
                                    back_string_end = 0
                                else:
                                    # Thời gian ngắt quá ngắn => nối dòng lại
                                    if temp_string == '':
                                        temp_start = margin_start
                                    temp_string += ' ' + current_text
                                    # print("Current text", current_text, "=====", temp_string)
                                    back_string_end = (current_end*2 + margin_end*3) / 5 - 0.2

                                flag_skip = True
                                index_skip = 0

                            elif temp_string == '':
                                temp_start = margin_start

                            # if idx > 46: break
                except IndexError as e:
                    # traceback.print_exc()
                    continue

        return results



def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)


def worker(idx, path, total):
    name = os.path.basename(path)

    vtt_path = os.path.join(path, name + '.vi.vtt')
    mp3_path = os.path.join(path, name + '.mp3')
    wav_path = os.path.join(path, name + '.wav')
    meta_path = os.path.join(path, 'meta.json')

    if (not os.path.exists(vtt_path)) or (not os.path.exists(mp3_path)):
        print(f'[{idx}/{total}] {path} ---> error')
        return

    print(f'[{idx}/{total}] {path}')
    # wav_dir = os.path.join(path, 'wav')
    wav_dir = os.path.join(path, 'wav')
    os.makedirs(wav_dir, exist_ok=True)

    if not os.path.exists(wav_path):
        # print('convert mp3 to wav')
        cmd = f'ffmpeg -y -i "{mp3_path}" -ac 1 -ar 16000 "{wav_path}" > /dev/null 2>&1 < /dev/null'
        os.system(cmd)  

    try:
        fs, data = wavfile.read(wav_path)
        # data, fs= sf.read(wav_path)
    except:
        print(f'[{idx}/{total}] {path} ---> read error')
        os.system(f'rm -rf {wav_path}')
    else:
        with open(f'{path}/{name}.json', 'w', encoding='utf8') as fp:
            for i, (start, end, text) in enumerate(read_line_vtt(vtt_path, meta_path)):
                # print("Start: {}, end: {}".format(start, end))
                # print("Text: {}".format(text.strip()))
                # print()
                # print("Raw time: {}".format(raw_time.strip()))
                data_tmp = data[int(fs * start) : int(fs * end)]
                name_tmp = os.path.join(wav_dir, f'{name}_{i}.wav')
                # sf.write(name_tmp, data_tmp, fs)
                wavfile.write(name_tmp, fs, data_tmp)
                meta = {
                    'audio_filepath': name_tmp,
                    'text': text.strip().lower(),
                    'duration': len(data_tmp)/fs,
                }
                json.dump(meta, fp, ensure_ascii=False)
                fp.write('\n')


        os.system(f'rm -rf {wav_path}')


if __name__ == '__main__':
    # path = '/home/dvhai/uUQziFLrtrk'
    # worker(path)
    root_dir = '/media/storage/hai/dataset'
    process_dir = [
        'datasets_18092021',
        # 'datasets_blv_anh_quan',
        # 'datasets_car_24h',
        # 'datasets_duong_dia_ly',
        # 'datasets_nguoc_dong_lich_su',
        # 'datasets_phe_phim',
        # 'datasets_thoisutoancanh_2',
        # 'datasets_vov_doc_truyen',
        # 'datasets_vov_lite',
        # 'datasets_vtcnow_2',
        # 'datasets_vtv2_2',
        # 'datasets_vtv24_2',
        # 'datasets_vtv4',
    ]
    # exclude_dir = open('exclude.txt').read().strip().split('\n')
    # print(exclude_dir)

    # paths = list(set([os.path.dirname(f) for d in process_dir 
    #                                      for f in recursive_walk(os.path.join(root_dir, d)) 
    #                                      if f.endswith('.mp3') and (os.path.basename(os.path.dirname(f)) not in exclude_dir)]))

    paths = list(set([os.path.dirname(f) for d in process_dir for f in recursive_walk(os.path.join(root_dir, d)) if f.endswith('.mp3')]))
    # print(len(paths))
    # print([f for f in paths[:10]])
    # path = paths[0]
    Parallel(n_jobs=10)(delayed(worker)(idx, path, len(paths)) for idx, path in enumerate(paths))