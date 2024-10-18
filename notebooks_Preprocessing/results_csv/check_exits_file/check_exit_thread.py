import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Đọc tệp CSV
metadata_check = pd.read_csv("/mnt/driver/STT_data/STT_dataset/64.Metadata_t5_64_63_..._57RDC/64.Metadata_t6__exits.csv")

# Hàm kiểm tra sự tồn tại của tệp
def check_exists(file_path):
    return os.path.exists(file_path)

# Sử dụng ThreadPoolExecutor để kiểm tra song song
with ThreadPoolExecutor() as executor:
    file_exists = list(tqdm(executor.map(check_exists, metadata_check['file']), total=len(metadata_check)))

# Thêm cột kiểm tra vào DataFrame
metadata_check['exists'] = file_exists

# Lọc các hàng có tệp tồn tại và không tồn tại
train_data_exists = metadata_check[metadata_check['exists']]
train_data_not_exists = metadata_check[~metadata_check['exists']]

# Chỉ giữ lại các cột cần thiết
columns_to_keep = ['file', 'text', 'duration']
train_data_exists = train_data_exists[columns_to_keep]
train_data_not_exists = train_data_not_exists[columns_to_keep]

# In ra các tệp tồn tại và không tồn tại
print("Files that exist:")
print(train_data_exists)
print("\nFiles that do not exist:")
print(train_data_not_exists)

# Lưu lại kết quả nếu cần
train_data_exists.to_csv('/home/pdnguyen/fast_confomer_finetun/finetune-fast-conformer/notebooks/results_csv/check_exits_file/out_csv/t6_exist.csv', index=False, encoding='utf-8')
train_data_not_exists.to_csv('/home/pdnguyen/fast_confomer_finetun/finetune-fast-conformer/notebooks/results_csv/check_exits_file/out_csv/t6_no_exist.csv', index=False, encoding='utf-8')
