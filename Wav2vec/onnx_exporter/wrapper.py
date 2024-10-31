import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import argparse
from transformers.trainer_utils import get_last_checkpoint
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor

class CMCWav2vec(nn.Module):

    def __init__(self, last_checkpoint, model_type='large', device='cpu'):
        super().__init__()
        self.processor = Preprocessor(model_type, device)
        self.classifier = Wav2Vec2ForCTC.from_pretrained(last_checkpoint).to(device)
        self.device = device

    def forward(self, x):
        input_values = self.processor(x)
        logits = self.classifier(input_values).logits
        return logits, self.processor.decode(logits)

class Preprocessor(nn.Module):

    def __init__(self, model_type='large', device='cpu'):
        super().__init__()

        tokenizer = Wav2Vec2CTCTokenizer(
            f'../vocab/vocab_{model_type}.json', 
            unk_token='<unk>', 
            pad_token='<pad>', 
            word_delimiter_token=' ' if model_type.lower() == 'large' else '|'
        )

        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, 
            sampling_rate=16000, 
            padding_value=0.0, 
            do_normalize=True, 
            return_attention_mask=False
        )
        self.processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.device = device


    def decode(self, logits):
        argmax_prediction = self.processor.batch_decode(torch.argmax(logits, dim=-1))
        return argmax_prediction[0]

    def forward(self, x):
        input_values = self.processor(x, sampling_rate=16000, return_tensors='pt', padding='longest').input_values.to(self.device)
        return input_values

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--model_type",
#         type=str,
#         choices=['base', 'large'],
#         default='large'
#     )
#     args = parser.parse_args()
#     args.device = 'cpu'

#     ###### 

#     onnx_model_name = "wav2vec.onnx"
#     ort_session = ort.InferenceSession(onnx_model_name)
#     input_torch_random = torch.randn(1, 16000 * 30)
#     outputs = ort_session.run(None, {'input': input_torch_random.numpy()})[0]
#     print(outputs.shape)

#     ###### 

#     save_dir = f'../output/wav2vec2-{args.model_type}-nguyenvulebinh'
#     last_checkpoint = get_last_checkpoint(save_dir)
#     print(f"last_checkpoint: {last_checkpoint}")
