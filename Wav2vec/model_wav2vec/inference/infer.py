import argparse
import warnings
warnings.filterwarnings('ignore')
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
import soundfile as sf
import torch
import tqdm
import os
import re
from utils.wer import wer

TMP_NAME = 'tmptmp.wav'
CMD_CONVERT = 'ffmpeg -nostats -loglevel 0 -y -i {} -ac 1 -ar 16000 {}'


def clean_text(string):
    text = re.sub(r'\_+', ' ', string)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.lower().strip()

def main():

    parser = argparse.ArgumentParser(description='Wav2Vec Inference')
    parser.add_argument('--model', type=str, default='jonatasgrosman/wav2vec2-large-xlsr-53-english')
    parser.add_argument('--tsv', type=str)
    parser.add_argument('--wav', type=str, default='ted.wav')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    if 0:
        device = torch.device(args.device)

        # greedy decoder
        processor_1 = Wav2Vec2Processor.from_pretrained(args.model) 
        
        # kenlm decoder
        processor_2 = Wav2Vec2ProcessorWithLM.from_pretrained(args.model) 
        
        # asr
        model = Wav2Vec2ForCTC.from_pretrained(args.model).to(device)

        def predict(speech, fs):
            # tokenizer
            input_values_1 = processor_1(speech, sampling_rate=fs, return_tensors='pt', padding='longest').input_values.to(device)
            input_values_2 = processor_2(speech, sampling_rate=fs, return_tensors='pt', padding='longest').input_values.to(device)
            
            assert torch.equal(input_values_1, input_values_2)
            # retrieve logits
            logits = model(input_values_1).logits
            return logits.cpu().detach()
    
        def infer_one(path):
            data, fs = sf.read(path)
            with torch.no_grad():
                logits = predict(data, fs)
                # greedy
                p = processor_1.batch_decode(torch.argmax(logits, dim=-1))
                # kenlm
                p_lm = processor_2.batch_decode(logits.numpy()).text
            return p[0].lower(), p_lm[0].lower()
            
        if args.tsv is None:
            speech, fs = sf.read(args.wav)
            with torch.no_grad():
                logits = predict(speech[:int(60*fs)], fs)
                # greedy
                p = processor_1.batch_decode(torch.argmax(logits, dim=-1))
                # kenlm
                p_lm = processor_2.batch_decode(logits.numpy()).text

            print('Greedy:', p[0])
            print()
            print('LM    :', p_lm[0])
            print()
        else:
            lines = open(args.tsv).read().strip().split('\n')[1:]
            # print(lines[:10])
            datadir = os.path.join(os.path.dirname(args.tsv), 'clips')
            with open('output.csv', 'w') as fp:
                for line in tqdm.tqdm(lines):
                    path, text = line.split('\t')[1:3]
                    path = os.path.join(datadir, path)
                    # print(path, text)
                    assert os.path.exists(path)
                    os.system(f'rm -rf {TMP_NAME}')
                    os.system(CMD_CONVERT.format(path, TMP_NAME))
                    assert os.path.exists(TMP_NAME)
                    p, p_lm = infer_one(TMP_NAME)
                    # print(p)
                    # print(p_lm)
                    # print()
                    print(f'{os.path.basename(path)},{clean_text(text)},{p},{p_lm}', file=fp)
    refs = []
    hyps = []            
    hyps_lm = []            
    for item in [line.split(',') for line in open('output.csv').read().strip().split('\n')]:
        filename, label, p, p_lm = item
        # if not filename.endswith('.wav'):
        #     continue

        if not isinstance(p, str):
            continue
        
        if not isinstance(p_lm, str):
            continue
        
        refs.append(label)
        hyps.append(p)
        hyps_lm.append(p_lm)
        
        # print(item)
        # break
    # print(refs)
    result = wer(refs, hyps) 
    print('WER:', result)

    result = wer(refs, hyps_lm) 
    print('WER LM:', result)

if __name__ == '__main__':
    main()
