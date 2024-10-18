import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm

# Đọc tệp metadata
metadata_check = pd.read_csv("/home/pdnguyen/fast_confomer_finetun/CSV_process_data/data_check_var.csv")

# Sử dụng tqdm để thêm thanh tiến trình
tqdm.pandas(desc="Checking file existence")

# Kiểm tra sự tồn tại của các tệp và thêm thanh tiến trình
train_data_exists = metadata_check[metadata_check['file'].progress_apply(os.path.exists)]

# Lấy các dòng không tồn tại bằng cách trừ các dòng tồn tại từ metadata_check
train_data_not_exists = metadata_check[~metadata_check.index.isin(train_data_exists.index)]

# In ra các dòng có tệp tồn tại
print("Files that exist:")
print(train_data_exists.shape)
train_data_exists.to_csv("/home/pdnguyen/fast_confomer_finetun/CSV_process_data/57_exits.csv", index=False)
# In ra các dòng không có tệp tồn tại
print("Files that do not exist:")
print(train_data_not_exists.shape)
train_data_not_exists.to_csv("/home/pdnguyen/fast_confomer_finetun/CSV_process_data/57_isnot_exits.csv", index=False)

