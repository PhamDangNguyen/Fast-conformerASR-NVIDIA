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

from wrapper import Preprocessor, CMCWav2vec
from util import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        choices=['base', 'large'],
        default='large'
    )
    args = parser.parse_args()
    args.device = 'cpu'

    filename, text, duration = get_random_example()
    speech_orig, fs = sf.read(filename)

    speech = padding(speech_orig, 30)

    preprocessor = Preprocessor(args.model_type, args.device)
    # print(preprocessor.processor.feature_extractor)
    # exit()

    ###### 
    if 1:
        onnx_model_name = "wav2vec.onnx"
        ort_session = ort.InferenceSession(onnx_model_name)
        # input_torch_random = torch.randn(1, 16000 * 30)
        preprocessed_speech = preprocessor(speech)
        # output_onnx = ort_session.run(None, {'modelInput': [speech]})[0]
        output_onnx = ort_session.run(None, {'input': preprocessed_speech.numpy()})[0]
        output_onnx = torch.tensor(output_onnx)
        print('output_onnx', output_onnx.shape)

    ###### 
    if 1:
        save_dir = f'../output/wav2vec2-{args.model_type}-nguyenvulebinh'
        last_checkpoint = get_last_checkpoint(save_dir)
        print(f"last_checkpoint: {last_checkpoint}")

        cmc_wav2vec = CMCWav2vec(last_checkpoint, args.model_type)
        # output_gt, pytorch_prediction = cmc_wav2vec([speech_orig])
        output_gt, pytorch_prediction = cmc_wav2vec(speech)
        print('output_gt', output_gt.shape)
    
    # print(torch.allclose(torch.tensor(output_onnx), output_gt, 1e-2))
    if output_onnx.shape == output_gt.shape:
        assert torch.allclose(output_onnx, output_gt, 0, 1e-3)
        print((output_onnx - output_gt).abs().sum())
        print((output_onnx - output_gt).abs().max())

    print(filename)
    print('label  :', text)
    print('pytorch:', pytorch_prediction)
    print('onnx   :', preprocessor.decode(output_onnx))