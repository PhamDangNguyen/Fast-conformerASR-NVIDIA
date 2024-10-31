from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
from transformers import Wav2Vec2Processor, WavLMForCTC
from transformers import Wav2Vec2ForCTC
import soundfile as sf
import numpy as np
import argparse
import torch
import random
import kenlm
import torch
import time
import tqdm
import json
import os
import re

from vad import segmentation

def main():
    filename = 'test/hop-Sap.wav'
    repo_name = 'patrickvonplaten/wavlm-libri-clean-100h-large'

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.device = torch.device(args.device)

    processor = Wav2Vec2Processor.from_pretrained(repo_name)
    model = WavLMForCTC.from_pretrained(repo_name).to(args.device)

    def predict(speech, fs):
        input_values = processor(speech, sampling_rate=fs, return_tensors='pt', padding='longest').input_values.to(args.device)
        logits = model(input_values).logits
        argmax_prediction = processor.batch_decode(torch.argmax(logits, dim=-1))
        return argmax_prediction[0]


    speech, fs = sf.read(filename)

    with open('hop-Sap-trans-2.txt', 'w') as fp:
        for (start, end) in segmentation(filename, min_duration=5, max_duration=20):
            data = speech[int(start*fs):int(end*fs)]
            print(start, end)
            p = predict(data, fs)
            print(f'{start:.02f},{end:.02f},{p}', file=fp)
            # break

if __name__ == '__main__':
    main()
