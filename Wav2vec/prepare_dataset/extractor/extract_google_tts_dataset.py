import os
import csv
import unicodedata
import soundfile as sf
from pydub import AudioSegment


# with open('../dataset_cmc_remove_duplicate/google_api_train.csv', 'w') as fp:
#     fp.write('file,text,duration\n')
#     f = open("/home/ndanh/STT_dataset/Google/info.csv")
#     csv_reader = csv.reader(f)
#     fields = next(csv_reader)
#     unique_line = set()
#     for line in csv_reader:
#         id,normalized_transcript,transcript = line[0],line[1],line[2]
#         if normalized_transcript == '':
#             continue
#         if normalized_transcript not in unique_line:
#             unique_line.add(normalized_transcript)
#             normalized_transcript = unicodedata.normalize("NFKC", normalized_transcript)
#             normalized_transcript = normalized_transcript.lower()
#             normalized_transcript = normalized_transcript.strip()
#             audio_path = os.path.join("/home/ndanh/STT_dataset/Google/audio_1_1ac_16k",f'{id}.wav')
#             audio, sr = sf.read(audio_path)
#             duration = audio.shape[0] / sr
#             print(f'{audio_path},{normalized_transcript},{duration}', file=fp)


sound1 = AudioSegment.from_file("/home/ndanh/STT_dataset/Google/audio_1_1ac_16k/13044.wav", format="wav")
sound2 = AudioSegment.from_file("/home/ndanh/STT_dataset/10.telesale/audio_telesale/tmduy/1577938020.181038_16k.wav/split_1/split_1.wav", format="wav")

# sound1 6 dB louder
# louder = sound1 + 6


# sound1, with sound2 appended (use louder instead of sound1 to append the louder version)
overlay = sound1.overlay(sound2, position=0)


# simple export
file_handle = overlay.export("output.mp3", format="mp3")


            