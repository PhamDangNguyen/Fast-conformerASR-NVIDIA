import pandas as pd
from pathlib import Path

file_need_filter = "/home/lmnguyen/Projects/cmc_stt/finetune-fast-conformer/val_need_filter.csv"
df_train =  pd.read_csv(file_need_filter)
df_train = df_train[(df_train["duration"] > 0.5) & (df_train["duration"] < 20)]
df_train["file"] = df_train["file"].apply(lambda x: x.replace("/home/ndanh/STT_dataset", "/home/ntdong/Data/STT_dataset"))
print(len(df_train["file"]))

#remove file 
# remove_file = open("/home/lmnguyen/Projects/cmc_stt/finetune-fast-conformer/notebooks/resources/results/final_filter_merge.txt").read().split("\n")
# mask_file = df_train["file"].isin(remove_file)
# df_train = df_train[~mask_file]
# print(len(df_train["file"]))

oov = open("/home/lmnguyen/Projects/cmc_stt/finetune-fast-conformer/notebooks/resources/results/final_filter_merge.txt").read().split("\n")
oov = [w.split("#") for w in oov]
dict_info = {}
for text in oov:
    try:
        dict_info[text[0].strip()] = text[1].strip()
    except:
        continue
print(len(dict_info))
print(dict_info)

for search_text_key, replacement in dict_info.items():
    mask = df_train["text"].str.contains(rf'\b{search_text_key}\b', regex=True, case=False, na=False)
    if replacement == "false":
        df_train = df_train[~mask] #remove rows has index mask = true in df_train
    elif replacement == "true":
        continue
        # print(search_text_key)
    else:
        df_train.loc[mask, "text"] = df_train[mask]["text"].str.replace(search_text_key,replacement)
        print(replacement)
        print(df_train[mask]["text"])

df_train.to_csv('/home/lmnguyen/Projects/cmc_stt/finetune-fast-conformer/notebooks/results_csv/origin_val_rms.csv', index=False)