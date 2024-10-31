import onnx
import onnxruntime as ort
import torch
import argparse
from transformers.trainer_utils import get_last_checkpoint
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
import random
import os
import soundfile as sf
import numpy as np
import tqdm

from wrapper import Preprocessor, CMCWav2vec
from util import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=['base', 'large'], default='large')
    parser.add_argument("--type", type=str, choices=['onnx', 'pytorch'])
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument("--csv", type=str)
    args = parser.parse_args()

    preprocessor = Preprocessor(args.model_type, args.device)
    
    if args.type == 'onnx':
        onnx_model_name = "wav2vec.onnx"
        onnx_model = ort.InferenceSession(onnx_model_name)
    elif args.type == 'pytorch':
        save_dir = f'../output/wav2vec2-{args.model_type}-nguyenvulebinh'
        last_checkpoint = get_last_checkpoint(save_dir)
        print(f"last_checkpoint: {last_checkpoint}")
        pytorch_model = CMCWav2vec(last_checkpoint, args.model_type, args.device)
    else:
        raise


    # filename, text, duration = get_random_example()
    # print(filename)
    # speech_orig, fs = sf.read(filename)

    # speech = padding(speech_orig, 30)



    output_name = f'{os.path.splitext(os.path.basename(args.csv))[0]}_{args.type}.csv'

    with open(output_name, 'w') as fp:
        lines = open(args.csv).read().strip().split('\n')[1:1001]
        for line in tqdm.tqdm(lines):
            path, text, duration = line.split(',')

            speech, fs = sf.read(path)
            speech = padding(speech, 30)

            if args.type == 'onnx':
                logits = onnx_model.run(None, {'input': preprocessor(speech).numpy()})[0]
                logits = torch.tensor(logits)
            else:
                logits = pytorch_model(speech)[0]

            print(f'{text},{preprocessor.decode(logits)}', file=fp)
