import torch
import torch.nn as nn
import argparse
from transformers.trainer_utils import get_last_checkpoint
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
# from transformers import Wav2Vec2FeatureExtractor
# from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel, build_ctcdecoder
import soundfile as sf
from statistics import mean
import math
import copy
import datetime


class CMCWav2vec(nn.Module):

    def __init__(self, last_checkpoint, model_type='large', device='cpu'):
        super().__init__()
        self.processor = Preprocessor(last_checkpoint,model_type, device)
        self.classifier = Wav2Vec2ForCTC.from_pretrained(last_checkpoint).to(device)
        self.classifier.eval()
        self.device = device
    def forward(self, x):
        
        input_values = self.processor(x)
        
        logits = self.classifier(input_values.to(self.device)).logits
        
        return logits, self.processor.decode(logits)

class Preprocessor(nn.Module):

    def __init__(self, last_checkpoint,model_type='large', device='cpu'):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(last_checkpoint)
        self.device = device


    def decode(self, logits):
        argmax_prediction = self.processor.batch_decode(torch.argmax(logits, dim=-1))
        return argmax_prediction[0]

    def forward(self, x):
        input_values = self.processor(x, sampling_rate=16000, return_tensors='pt', padding='longest').input_values.to(self.device)
        return input_values
    
def load_all_model(last_checkpoint):
    # device = "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pytorch_model = CMCWav2vec(last_checkpoint, 'large', device=device)
    return pytorch_model,device

def infer(audio_path,pytorch_model):
    audio,sr = sf.read(audio_path, dtype='float32')
    
    
    logits = pytorch_model(audio)[0]
    

    argmax_prediction = pytorch_model.processor.decode(logits)

    return argmax_prediction

if __name__ == "__main__":
    save_dir = f'output/wav2vec2-large-nguyenvulebinh-original/'
    # last_checkpoint = get_last_checkpoint(save_dir)
    # print(f"last_checkpoint: {last_checkpoint}")
    last_checkpoint = "/home/ndanh/asr-wav2vec/output/checkpoint-17120000_13340000"
    pytorch_model,device = load_all_model(last_checkpoint)
    audio_path = "1.wav"
    results = infer(audio_path,pytorch_model)
    print(results)
