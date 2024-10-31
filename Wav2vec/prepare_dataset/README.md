## I. Introduction

There are 4 types of dataset:

- Dataset public
- Dataset annotated
- Dataset crawl from tiktok, youtube
- Dataset raw (without transcription)

For detail, access project sheet: [Link](https://docs.google.com/spreadsheets/d/1Cu2afizBlADK1G-aU05mfB946Ho7ZOHz0LKEYC58ZYY/edit?usp=sharing)

## II. Preprocess public dataset

There are some steps to create dataset ready to train model STT

Step 1: Create config file original dataset: /extractor
Step 2: Create csv of each dataset: /generate
Step 3: Resample original audio, Augment audio and create corresponding csv files: /augment
Step 4: Export, generate and split train/dev/test from all config of all dataset
- Utils: /utils/
    - Check error path
    - Check exist audio path

#### Step 1: Convert audio dataset to 16khz


Each dataset has different characteristic, so each dataset must its own preprocess Folder: prepare_dataset/converter
Example:

```commandline
python -m prepare_dataset.converter.1.convert_open_fpt
```

#### Step 2: Create Config file

Store all metadata of audio file into config file to "config" folder Folder: prepare_dataset/extractor Example:

```commandline
python -m prepare_dataset.extractor.1.extract_open_fpt
```

#### Step 3: Create train/dev/test file from config files

Lưu ý: repo này đã tồn tại lâu nên code dưới chỉ mục đích tham khảo

```commandline
python generate_dataset.py
python export.py
```

## III. Kiểm tra chất lượng dữ liệu

<b>Chi tiết xem tại repo: [Voice preprocessing](https://gitlab.com/ngtiendong/voice-preprocessing)
</b>

Các bước kiểm tra bao gồm:

- Check có số hay không
- Check xem có thể dùng model hiện tại, để sử dụng các transcript có số hay không
- Check chính tả, kiểm tra các từ có trong từ điển en, vi không. Nếu không, cần kiểm tra bằng tay
- Sử dụng force alignment:
    - Lựa chọn xem có cắt đầu cuối lại, tránh trường hợp mất từ đầu cuối hay không
    - Trong 1 câu, có thể chỉ chọn vài từ có confident cao trong đó kèm ở giữa

- Lọc dataset chất lượng:
To determine whether data is good enough, there are 4 steps to take out:
  
Step 1: Infer trained model ASR:
- Because model train in large dataset quality so that accuracy is usually more than 80%
- Input: csv file contains audio path and transcript corresponding
- Output: csv file result predict score Step 2: Remove hard sample Step 3: Create new dataset Step 4: Gen csv data
      train

```commandline
python -m prepare_dataset.correct_dataset # should check detail 4 steps above
```

## IV. Augment dataset
After having clean data, need to apply some technique augmentation
We apply several techniques to augment dataset:
### ALERT: Code is updated in long time, so check "dataset" path load in each code file carefully to get newest dataset

#### IV.1 Standard augment technique:
- Rir
- Codec
- Hard noise of full audio

```commandline
python -m augment.1_standard_augment
```

#### IV.2 Add short noise: noise in some moment in audio
```commandline
python -m augment.2_short_noise_augment
```

#### IV.3 Concat audio from one speaker to multi speaker:
```commandline
python -m augment.3_concat_data_stt
```

#### IV.4 Some augments: GaussianNoise, TimeStretch, PitchShift, Shift 
```commandline
python -m augment.4_new_augment
```

#### IV.5 Some augments: GaussianNoise, TimeStretch, PitchShift, Shift 
```commandline
python -m augment.4_new_augment
python -m augment.4_duyanh_augment
```

## V. Split train/test






