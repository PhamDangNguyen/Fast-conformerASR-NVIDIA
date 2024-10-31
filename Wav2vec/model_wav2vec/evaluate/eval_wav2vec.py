import warnings

warnings.filterwarnings('ignore')
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
import soundfile as sf
import kenlm
import torch
import tqdm
import json
import os
import re
import pandas as pd

import sys

sys.path.insert(0, '/home/dvhai/code/voice-preprocessing/denoiser_stream')

from utils.wer import get_wer
# from pretrained import add_model_flags

import argparse

VOICEFIXER_CMD = 'voicefixer --infile "{}" --outfile "{}" --mode 1 --silent'
CMD_CONVERT_1CH = 'ffmpeg -y -i "{}" -ac 1 -ar 16000 "{}" > /dev/null 2>&1 < /dev/null'


def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)


def read_config(filename):
    with open(filename) as fid:
        return [json.loads(l) for l in fid]


def main():
    parser = argparse.ArgumentParser()

    # add_model_flags(parser)
    parser.add_argument("--sample_rate", default=16000, type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("-t", "--num_threads", type=int)
    parser.add_argument("-f", "--num_frames", type=int, default=1)
    parser.add_argument("--dataset", default="common_voice")
    parser.add_argument("--demucs", action="store_true")
    parser.add_argument("--voicefixer", action="store_true")
    parser.add_argument("--model_type", default="large")
    args = parser.parse_args()

    assert not (args.demucs and args.voicefixer)

    lm_path = './models/lm/vi_lm_4grams.bin'
    # lm_path = './models/lm/vi_en_4gram.binary'
    # lm_path = './models/lm/vi_4gram.binary'
    # lm_path = './models/lm/vi_4gram_20m.binary'
    assert os.path.exists(args.dataset)
    testcase = os.path.basename(args.dataset).split('.')[0]

    if args.demucs:
        from pretrained import get_model
        demucs = get_model(args).to(args.device)

        if args.dns64:
            output_name = f'{testcase}_infer_{args.model_type}_dns64.csv'
        else:
            output_name = f'{testcase}_infer_{args.model_type}_dns48.csv'
    elif args.voicefixer:
        from voicefixer import VoiceFixer
        voicefixer_model = VoiceFixer()
        output_name = f'{testcase}_infer_{args.model_type}_voicefixer.csv'
        # print(voicefixer_model)
        # exit()
    else:
        output_name = f'{testcase}_infer_{args.model_type}_False.csv'

    save_dir = f'output/wav2vec2-{args.model_type}-nguyenvulebinh'
    # save_dir = '/home/dvhai/ckpt/base_full_noise_en_wer_15.5'

    # load model and tokenizer
    tokenizer = Wav2Vec2CTCTokenizer(
        f'./vocab/vocab_{args.model_type}.json',
        unk_token='<unk>',
        pad_token='<pad>',
        word_delimiter_token=' ' if args.model_type.lower() == 'large' else '|'
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False
    )

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    last_checkpoint = get_last_checkpoint(save_dir)
    print(f"last_checkpoint: {last_checkpoint}")

    model = Wav2Vec2ForCTC.from_pretrained(last_checkpoint).to(args.device)

    # language model

    # beam search
    def get_decoder_ngram_model(tokenizer, ngram_lm_path):
        vocab_dict = tokenizer.get_vocab()
        sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
        vocab = [x[1] for x in sort_vocab][:-1]
        vocab_list = vocab
        # print(vocab_list)
        # print(len(vocab_list), tokenizer.unk_token_id, tokenizer.word_delimiter_token_id)
        # vocab_list[tokenizer.unk_token_id] = ""
        vocab_list[tokenizer.word_delimiter_token_id] = " "
        alphabet = Alphabet.build_alphabet(vocab_list)
        lm_model = kenlm.Model(ngram_lm_path)
        decoder = BeamSearchDecoderCTC(alphabet, language_model=LanguageModel(lm_model))
        return decoder

    ngram_lm_model = get_decoder_ngram_model(processor.tokenizer, lm_path)

    def predict(speech, fs, return_logits=False):
        # tokenizer
        input_values = processor(speech, sampling_rate=fs, return_tensors='pt', padding='longest').input_values.to(
            args.device)
        # retrieve logits
        logits = model(input_values).logits
        # argmax
        argmax_prediction = processor.batch_decode(torch.argmax(logits, dim=-1))
        beam_search_output = ngram_lm_model.decode(logits.cpu().detach().numpy().squeeze(), beam_width=500)
        # beam_search_output = ''
        if return_logits:
            return argmax_prediction[0], beam_search_output, logits

        return argmax_prediction[0], beam_search_output

    with open(output_name, 'w', encoding='utf8') as fp:
        fp.write('filename,label,predict\n')

        print('[+] Processing', args.dataset)
        bar = tqdm.tqdm(pd.read_csv(args.dataset, header=None).values)
        e = 0
        for item in bar:
            filename, label = item[0], item[1]
            if not filename.endswith('.wav'):
                continue
            speech, fs = sf.read(filename)

            if args.demucs:
                speech = torch.tensor(speech).float().to(args.device)
                speech = demucs(speech[None])[0, 0]

            if args.voicefixer:
                tmp_name = 'output_voicefixer_tmp.wav'
                tmp_16k_name = 'output_voicefixer_16k_tmp.wav'

                os.system(f'rm -rf {tmp_name}')
                os.system(f'rm -rf {tmp_16k_name}')

                # cmd = VOICEFIXER_CMD.format(filename, tmp_name)
                # print(cmd)
                # os.system(cmd)
                voicefixer_model.restore(input=filename, output=tmp_name, cuda=args.device, mode=0)
                assert os.path.exists(tmp_name)

                os.system(CMD_CONVERT_1CH.format(tmp_name, tmp_16k_name))
                assert os.path.exists(tmp_16k_name)

                speech, fs = sf.read(tmp_16k_name)
                assert fs == 16000
                # exit()

            try:
                p, p_lm = predict(speech, fs)
            except KeyboardInterrupt:
                break
            except:
                # raise
                e += 1
                bar.set_postfix(error=e)
            else:
                print('label:', label)
                print('p_lm :', p_lm)
                print()
                fp.write(f'{filename},{label},{p_lm}\n')

    """
    MAIN CALCULATE WER 
    """
    get_wer(output_name)


if __name__ == '__main__':
    main()
    # output_name = 'common_voice_infer.csv'
    # get_wer_common_voice(output_name)
