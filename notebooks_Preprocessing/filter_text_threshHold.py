from tone_normalization import normalize_vn_tone
import pandas as pd
from pathlib import Path
dfs = []
cmc_dir = "/home/lmnguyen/Projects/cmc_stt/asr-wav2vec-bartpho/datasets/final_correct_dataset_cmc_filter"
for file in Path(cmc_dir).rglob("*_train.csv"):
    dfs.append(pd.read_csv(file))

df_train = pd.concat(dfs)
df_train = df_train[(df_train["duration"] > 1) & (df_train["duration"] < 20)]
df_train["file"] = df_train["file"].apply(lambda x: x.replace("/home/ndanh/STT_dataset", "/home/ntdong/Data/STT_dataset"))

df_train["text"] = df_train["text"].apply(lambda x: normalize_vn_tone(x))

all_texts = df_train["text"].tolist()
all_words = set([word for text in all_texts for word in text.split()])

vn_words_1 = open("/home/lmnguyen/Projects/cmc_stt/finetune-fast-conformer/notebooks/resources/vi_words_1.txt").read().split("\n")
vn_words_2 = open("/home/lmnguyen/Projects/cmc_stt/finetune-fast-conformer/notebooks/resources/vi_words_2.txt").read().split("\n")
vn_words_3 = open("/home/lmnguyen/Projects/cmc_stt/finetune-fast-conformer/notebooks/resources/vi_words_3.txt").read().split("\n")
en_words_1 = open("/home/lmnguyen/Projects/cmc_stt/finetune-fast-conformer/notebooks/resources/en_words_1.txt").read().split("\n")
en_words_2 = open("/home/lmnguyen/Projects/cmc_stt/finetune-fast-conformer/notebooks/resources/en_words_2.txt").read().split("\n")
en_words_2 = [w.split()[0] for w in en_words_2]
vn_words_all = set(vn_words_1 + vn_words_2 + vn_words_3)
en_words_all = set(en_words_1 + en_words_2)
print(len(en_words_all),len(vn_words_all))
vn_words_all = set([normalize_vn_tone(x) for x in vn_words_all])
en_words_all = set([normalize_vn_tone(x) for x in en_words_all])
valid_words = vn_words_all.union(en_words_all)
invalid_words = all_words.difference(valid_words)
print("len of invalid_words = ",len(invalid_words))



invalid_text = open("/home/lmnguyen/Projects/cmc_stt/finetune-fast-conformer/notebooks/resources/results/invalid_words.txt").read().split("\n")
text_filter = []
text_remove = []
info_text = []
for text in invalid_text:
    wav_file = df_train[df_train["text"].str.contains(rf'\b{text}\b', regex=True, case=False, na=False)]
    print(len(wav_file))
    if len(wav_file) < 5:
        text_remove.extend(wav_file["file"].tolist())
    else:
        text_filter.append(text)
    in4 = (text,len(wav_file))
    info_text.append(in4)

with open('/home/lmnguyen/Projects/cmc_stt/finetune-fast-conformer/notebooks/resources/results/final_text_filter.txt', 'w', encoding='utf-8') as file:
    # Ghi từng phần tử của mảng vào tệp, mỗi phần tử trên một dòng
    for text in text_filter:
        file.write(str(text) + '\n')
with open('/home/lmnguyen/Projects/cmc_stt/finetune-fast-conformer/notebooks/resources/results/final_text_remove.txt', 'w', encoding='utf-8') as file:
    # Ghi từng phần tử của mảng vào tệp, mỗi phần tử trên một dòng
    for text in text_remove:
        file.write(str(text) + '\n')

with open('/home/lmnguyen/Projects/cmc_stt/finetune-fast-conformer/notebooks/resources/results/info_text.txt', 'w', encoding='utf-8') as file:
    # Ghi từng phần tử của mảng vào tệp, mỗi phần tử trên một dòng
    for text in info_text:
        file.write(str(text) + '\n')