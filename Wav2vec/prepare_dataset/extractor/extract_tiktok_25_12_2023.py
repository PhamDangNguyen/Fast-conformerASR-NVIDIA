import csv
import os
import pandas as pd
import math
from sklearn.utils import shuffle

def recursive_walk(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".csv") and "error" not in file:
                yield(os.path.join(root,file))

csv_files = recursive_walk("/home/ndanh/STT_dataset/Tiktok_20_12_2023")
df = pd.concat([pd.read_csv(f) for f in csv_files], axis=0, ignore_index=True)
num_row = len(df)
num_val = math.floor(0.2*num_row)
df = shuffle(df)
# df.to_csv("test.csv",index=False)
df_val = df[:num_val]
df_train = df[num_val:]
df_train.to_csv("/home/ndanh/asr-wav2vec/final_correct_dataset_cmc_filter/25_12_2023_Tiktok_train.csv",index=False)
df_val.to_csv("/home/ndanh/asr-wav2vec/final_correct_dataset_cmc_filter/25_12_2023_Tiktok_test.csv",index=False)