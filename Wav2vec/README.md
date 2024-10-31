# I. Introduction
All about research in Speech Recognition
It contains:
- Prepare dataset
- Onnx convert
- Train/eval wav2vec

# II. Step 1: Prepare Dataset
Có các bước sau đây để tiền xử lý dữ liệu, tất cả trong folder "/prepare_dataset"
- Xử lý dữ liệu các tập dataset public
- Kiểm tra chất lượng dữ liệu
  - Kiểm tra có thừa thiếu audio so với transcript hay không
  - Lọc các transcript không thoả mãn
- Các biện pháp giúp tận dụng dữ liệu: 
  - Chuyển đổi số sang chữ số
  - Gán lại các text dạng tiếng anh
- Tạo tập train/dev/test cho model
- Filter dữ liệu và đếm 

# III. Step 2: Training Wav2vec
### 1. Config
- Save dir
- Tokenizer
- From pretrained
- training_args (thay đổi batch, epoch)

### 2. Train pretrain model
Huấn luyện pretrain unsupervised wav2vec
```bash
python -m wav2vec.train_large_pretrain
```

### 3. Finetune model
```commandline
python -m wav2vec.train_base # Train model base
python -m wav2vec.train_large # Train model large
```

### 4. Inference
```commandline
python -m wav2vec.infer
```

# IV. Deployment
### 1. Convert onnx & quantization
```commandline
python -m onnx_exporter.convert_torch_to_onnx
```

### 2. Evaluation: check WER
```commandline
python -m evaluate.eval_wav2vec.py --dataset /path/to/csv_eval_file
# Eg: python -m evaluate.eval_wav2vec --dataset "final_correct_dataset_cmc_filter/2022_12_cleaned_cmc_val.csv"
```

### 3. Other
Log all information of each dataset: file, hours, ...
```commandline
python -m utils.enumerate_dataset
```
