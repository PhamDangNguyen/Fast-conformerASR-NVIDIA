import json

input_file = "/home/pdnguyen/fast_confomer_finetun/Fast-conformerASR-NVIDIA/valid.json"
output_file = "/home/pdnguyen/fast_confomer_finetun/Fast-conformerASR-NVIDIA/metadata_train/validcv.json"

# Đọc dữ liệu từ file JSON đầu vào
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Chuyển đổi dữ liệu thành định dạng mới
new_data = []
for item in data:
    new_item = {
        "audio_filepath": item["name"],
        "duration": item["duration"],
        "text": item["transcript"]
    }
    new_data.append(new_item)

# Ghi dữ liệu mới ra file JSON đầu ra
with open(output_file, "w", encoding="utf-8") as f:
    for item in new_data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")  # Thêm dòng mới sau mỗi đối tượng JSON

print("Đã chuyển đổi xong dữ liệu.")