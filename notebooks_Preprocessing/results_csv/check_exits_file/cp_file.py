import tqdm 
import os
#cac lenh can thuc hien
move_giong_bac = ["/mnt/driver/STT_data/New_STT_Data/Tiktok/giong_bac/2024/7"]
move_giong_nam = ["/mnt/driver/STT_data/New_STT_Data/Tiktok/giong_nam/2024/5","/mnt/driver/STT_data/New_STT_Data/Tiktok/giong_nam/2024/6","/mnt/driver/STT_data/New_STT_Data/Tiktok/giong_nam/2024/7/5"]
move_giong_trung = ["/mnt/driver/STT_data/New_STT_Data/Tiktok/giong_trung/2024/5","/mnt/driver/STT_data/New_STT_Data/Tiktok/giong_trung/2024/6"]
target_dir = ["/mnt/driver/STT_data/STT_dataset/65.Tiktok_T7/giong_bac/2024","/mnt/driver/STT_data/STT_dataset/65.Tiktok_T7/giong_nam/2024","/mnt/driver/STT_data/STT_dataset/65.Tiktok_T7/giong_trung/2024"]

arr = [move_giong_bac,move_giong_nam,move_giong_trung]

for index,path in tqdm.tqdm(enumerate(target_dir)):
    for i in arr[index]:
        cmd = f"cp -r {i} {path}"
        os.system(cmd)
        print(f"da cp thu muc {i} toi thu muc {path}")