from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2Processor

from typing import Any, Dict, List, Optional, Union
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
import soundfile as sf
import pandas as pd
import torch
import os
import unicodedata
import numpy as np


def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)


class CustomDataset(Dataset):

    def __init__(self, mode='train', processor=None):
        # Update 1/2024: /csv_train_test
        csv_files = [f for f in recursive_walk(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset/final_correct_dataset_cmc_filter')) \
                     if f.endswith(f'_{mode}.csv')]
        print(csv_files)

        self.data = pd.concat([pd.read_csv(f) for f in csv_files], axis=0, ignore_index=True)

        if mode == 'train':
            self.data = self.data[1 < self.data['duration']].reset_index(drop=True)
            self.data = self.data[self.data['duration'] <= 20].reset_index(drop=True)

        if mode == 'val':
            self.data = self.data[1 < self.data['duration']].reset_index(drop=True)
            self.data = self.data[self.data['duration'] <= 20].reset_index(drop=True)

        self.processor = processor

    def __len__(self):
        return len(self.data)

    def speech_file_to_array_fn(self, batch):
        speech_array, sampling_rate = sf.read(batch['file'])
        if len(speech_array.shape) == 2:
            speech_array = np.squeeze(np.mean(speech_array, axis=1))

        batch['speech'] = speech_array
        batch['sampling_rate'] = sampling_rate
        text = batch['text'].lower()
        text = unicodedata.normalize("NFKC", text)
        text = text.strip()
        batch['target_text'] = text
        return batch

    def prepare_dataset(self, batch):
        batch['input_values'] = self.processor(batch['speech'], sampling_rate=batch['sampling_rate']).input_values[
            0].tolist()
        with self.processor.as_target_processor():
            batch['labels'] = self.processor(batch['target_text']).input_ids
        return batch

    def __getitem__(self, idx):
        # try:
        #     batch = self.data.iloc[idx].copy()
        # except:
        #     print('1', idx, '--------->', batch['file'])
        #     raise

        # try:
        #     batch = batch.to_dict()
        # except:
        #     print('2', idx, '--------->', batch['file'])
        #     raise
        # try:
        #     batch = self.speech_file_to_array_fn(batch)
        # except:
        #     print('3', idx, '--------->', batch['file'])
        #     raise

        # try:
        #     batch = self.prepare_dataset(batch)
        # except:
        #     print('4', idx, '--------->', batch['file'])
        #     raise
        batch = self.data.iloc[idx].copy()
        batch = batch.to_dict()
        if os.path.exists(batch['file']):
            batch['path'] = batch['file']
        elif os.path.exists(batch['path']):
            batch['file'] = batch['path']
        else:
            print(batch['file'])
            print(batch['file'])
            print(batch)
            print()
            exit()
        # print(batch['file'])
        batch = self.speech_file_to_array_fn(batch)
        batch = self.prepare_dataset(batch)
        return batch


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{'input_values': feature['input_values']} for feature in features]
        label_features = [{'input_ids': feature['labels']} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors='pt',
            )
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch['labels'] = labels
        return batch


if __name__ == '__main__':
    tokenizer = Wav2Vec2CTCTokenizer(
        './dataset/vocab.json',
        unk_token='[UNK]',
        pad_token='[PAD]',
        word_delimiter_token=' '
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False
    )

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # train_dataset = CustomDataset('train', processor)
    # train_loader = DataLoader(train_dataset, batch_size=10)

    # for idx, batch in enumerate(train_loader):
    #     print(idx)

    tokenizer.save_pretrained("./models/wav2vec2/")
