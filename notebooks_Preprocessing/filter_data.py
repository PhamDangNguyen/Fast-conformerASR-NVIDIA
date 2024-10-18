import pandas as pd
from pathlib import Path
import os
data_augment = pd.read_csv("/home/pdnguyen/fast_confomer_finetun/finetune-fast-conformer/notebooks/results_csv/augment_0_2_all_0_7_lound_speed.csv")
data_augment = data_augment.sample(frac=0.7, random_state=1).reset_index(drop=True)
augment_lound_speed = pd.read_csv("/home/pdnguyen/fast_confomer_finetun/finetune-fast-conformer/notebooks/results_csv/augment_0_2_all_0_7_lound_speed_train_2k1hours.csv")
augment_lound_speed = augment_lound_speed.sample(frac=0.3, random_state=1).reset_index(drop=True)

data_medical = pd.read_csv("/mnt/driver/STT_data/STT_dataset/64.VietMed/metadata_normalize_final.csv")
data_clean = pd.read_csv("/home/pdnguyen/fast_confomer_finetun/finetune-fast-conformer/notebooks/results_csv/2023_2024_origin_data_clean.csv")
data_tele_sale = pd.read_csv("/mnt/driver/STT_data/STT_dataset/62.Tele_extra_augment/metadata.csv")
data_augment_10_dir = pd.read_csv("/mnt/driver/pdnguyen/studen_annoted/csv/check_to_remove/augment_tele_10dir.csv")

data_augment_tele_add = pd.read_csv("/home/pdnguyen/fast_confomer_finetun/finetune-fast-conformer/notebooks/results_csv/metadata_train_tele_them_7_5_2024.csv")

dir_62_tele = pd.read_csv("/mnt/driver/STT_data/STT_dataset/62.Tele_extra_augment/metadata.csv")

tiktok_new_100h = pd.read_csv("/mnt/tiktok_extral.csv")

train_data = pd.concat([data_augment,augment_lound_speed,data_medical,data_clean,data_tele_sale,data_augment_10_dir,data_augment_tele_add,dir_62_tele,tiktok_new_100h])
train_data = train_data.sample(frac=1, random_state=1).reset_index(drop=True)

train_data = train_data.drop_duplicates(subset='file', keep='first')


total_time_minutes = train_data['duration'].sum()
print(total_time_minutes/3600)
print("bat dau qua trinh filter di ngu thoi")
train_data_exists = train_data[train_data['file'].apply(os.path.exists)]
train_data_exists.to_csv('/home/pdnguyen/fast_confomer_finetun/finetune-fast-conformer/notebooks/results_csv/data_OT_demo.csv', index=False, encoding='utf-8')