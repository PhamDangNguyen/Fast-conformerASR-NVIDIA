import os
import torchaudio
import glob
import json
import unicodedata
import csv
import re
from vietnam_number import w2n,n2w

def myFunc(e):
    return e[0]

def check_duplicate_id(audio_dir):
    total_audios = 0
    unique_ids = set()
    for user_id in os.listdir(audio_dir):
        num_ids = os.listdir(f'{audio_dir}/{user_id}')
        total_audios += len(num_ids)
        for id_ in num_ids:
            unique_ids.add(id_)
    
    unique_ids = list(unique_ids)
    assert total_audios == len(unique_ids)
    return total_audios

def convert_1ac_16khz(audio_dir):
    for root,dirs,files in os.walk(audio_dir):
        for file_ in files:
            if file_.endswith(".mp3"):
                id_ = os.path.splitext(os.path.basename(file_))[0]
                i = os.path.join(root,file_)
                o = i[:-4] + "_1ac_16khz.wav"
                cmd = f'ffmpeg -y -i "{i}" -ac 1 -ar 16000 -sample_fmt s16 -b:a 256k -resampler soxr "{o}" > /dev/null 2>&1 < /dev/null'
                os.system(cmd)

def find_audio_path_with_id(audio_dir):
    results = {}
    for root,dirs,files in os.walk(audio_dir):
        for file_ in files:
            if file_.endswith("_1ac_16khz.wav"):
                id_ = os.path.splitext(os.path.basename(file_))[0][:-10]
                results[id_] = os.path.join(root,file_)
    return results

def norm_sentence(sentence):
    parts = sentence.split(" ")
    norm_sentence = ""
    
    for part in parts:
        if not part.isalpha():
            if int(part) > 20:
                return None
            else:
                part = n2w(part)
            
        norm_sentence = norm_sentence + part
        norm_sentence = norm_sentence + " "
    norm_sentence = norm_sentence.strip()
    return norm_sentence


if __name__ == '__main__':
    total_audios = check_duplicate_id('/home/ndanh/STT_dataset/nghiem_thu_ytb_2022_02/Raw')
    convert_1ac_16khz('/home/ndanh/STT_dataset/nghiem_thu_ytb_2022_02/Raw')
    id_2_path = find_audio_path_with_id('/home/ndanh/STT_dataset/nghiem_thu_ytb_2022_02/Raw')
    assert total_audios == len(list(id_2_path.keys()))
    print(f'Total number of audio: {total_audios}')
    label_dir = '/home/ndanh/STT_dataset/nghiem_thu_ytb_2022_02/Result'
    labels = {}
    for json_file in glob.glob(f'{label_dir}/*.json'):
        with open(json_file) as f:
            data = json.load(f)
            for audio_annotation in data:
                original_filename = audio_annotation['original_filename']
                segmentations = audio_annotation['segmentations']
                
                if original_filename not in list(id_2_path.keys()):
                    continue

                labels[original_filename] = []

    for json_file in glob.glob(f'{label_dir}/*.json'):
        with open(json_file) as f:
            data = json.load(f)
            for audio_annotation in data:
                original_filename = audio_annotation['original_filename']
                segmentations = audio_annotation['segmentations']
                
                if original_filename not in list(id_2_path.keys()):
                    continue

                for segmentation in segmentations:
                    start_time = segmentation['start_time']
                    end_time = segmentation['end_time']
                    transcription = segmentation['transcription']
                    if transcription:
                        transcription = transcription.replace("\n","")
                        transcription = transcription.replace("\n","")
                        transcription = transcription.replace("\r","")
                        transcription = transcription.lower()
                        transcription = transcription.strip()
                        transcription = unicodedata.normalize("NFKC", transcription)
                        transcription = transcription.replace("\"","")
                        transcription = re.sub(r'[^\w\s]', '', transcription)
                        transcription = transcription.replace("  "," ")
                        transcription = transcription.replace("  "," ")
                        if [start_time,end_time,transcription] not in labels[original_filename]:
                            labels[original_filename].append([start_time,end_time,transcription])
    
    for filename_ in labels.keys():
        labels[filename_] = list(labels[filename_])
        labels[filename_].sort(key=myFunc)
    
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    g = open("data_cmc_vad_test.csv",'w',newline='',encoding = 'utf8')
    csv_writer = csv.writer(g)
    csv_writer.writerow(['file','text','duration'])
    k = open("data_cmc_vad_error.csv",'w',newline='',encoding = 'utf8')
    csv_writer_k = csv.writer(k)
    csv_writer_k.writerow(['text'])
    for filename_ in labels.keys():
        if not os.path.exists(f'/home/ndanh/STT_dataset/nghiem_thu_ytb_2022_02/split/{filename_}'):
            os.makedirs(f'/home/ndanh/STT_dataset/nghiem_thu_ytb_2022_02/split/{filename_}',exist_ok=True)
        audio_path = id_2_path[filename_]
        audio,sr = torchaudio.load(audio_path)
        all_segmentations = labels[filename_]
        count = 0
        for segmentation in all_segmentations:
            start_time = segmentation[0]
            end_time = segmentation[1]
            start_ = max(int(start_time * 16000),0)
            end_ = min(int(end_time * 16000),audio.shape[1])
            transcription = segmentation[2]
            try:
                norm_transcription = norm_sentence(transcription)
                if norm_transcription is not None:
                    duration = end_time - start_time
                    audio_cut = audio[:,start_:end_]
                    torchaudio.save(f'/home/ndanh/STT_dataset/nghiem_thu_ytb_2022_02/split/{filename_}/{count}.wav', audio_cut, sr, encoding="PCM_S", bits_per_sample=16)
                    csv_writer.writerow([f'/home/ndanh/STT_dataset/nghiem_thu_ytb_2022_02/split/{filename_}/{count}.wav',norm_transcription,duration])
                    count += 1
            except:
               print(f'Sentence: "{transcription}" contain strange word')
               csv_writer_k.writerow([transcription])
            
    print(f"Finishing process")