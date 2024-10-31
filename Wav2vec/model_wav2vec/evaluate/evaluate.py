from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch

from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
import kenlm
import os

if __name__ == '__main__':

    # /media/storage/hai/dataset/2.tts_big_vinno/wav/05344_new.wav,người mẹ hàn quốc chặn xe máy leo vỉa hè hà nội,3.9009375
    FILENAME = 'audios/2.wav'


    cmd = f'ffmpeg -y -i "{FILENAME}" -ac 1 -ar 16000 "audios/test.wav" > /dev/null 2>&1 < /dev/null'
    os.system(cmd)
    FILENAME = 'audios/test.wav'
    
    # load model and tokenizer
    processor = Wav2Vec2Processor.from_pretrained('./models/eval/')
    model = Wav2Vec2ForCTC.from_pretrained('./models/eval/')

    processor.tokenizer.save_vocabulary('./output')

    # language model
    lm_path = '/home/dvhai/code/wav2vec2/models/lm/vi_lm_4grams.bin' 

    # define function to read in sound file
    def map_to_array(batch):
        speech, sr = sf.read(batch['file'])
        print(speech.shape, sr)
        batch['speech'] = speech
        return batch

    # load dummy dataset and read soundfiles
    ds = map_to_array({'file': FILENAME})

    # tokenize
    input_values = processor(ds['speech'], return_tensors='pt', padding='longest').input_values  # Batch size 1

    # retrieve logits
    logits = model(input_values).logits

    # argmax
    argmax_prediction = processor.batch_decode(torch.argmax(logits, dim=-1))

    # beam search
    def get_decoder_ngram_model(tokenizer, ngram_lm_path):
        vocab_dict = tokenizer.get_vocab()
        sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
        vocab = [x[1] for x in sort_vocab][:-2]
        vocab_list = vocab
        # replace special characters
        #vocab_list[tokenizer.pad_token_id] = ""
        vocab_list[tokenizer.unk_token_id] = ""
        #print(len(vocab_list), vocab_list)
        vocab_list[tokenizer.word_delimiter_token_id] = " "
        #print(f"Vocab size: {len(vocab)}")
        # specify ctc blank char index, since conventially it is the last entry of the logit matrix
        #print({n: c for n, c in enumerate(vocab_list)})
        alphabet = Alphabet.build_alphabet(vocab_list)
        lm_model = kenlm.Model(ngram_lm_path)
        decoder = BeamSearchDecoderCTC(alphabet, language_model=LanguageModel(lm_model))

        return decoder

    ngram_lm_model = get_decoder_ngram_model(processor.tokenizer, lm_path)
    logits_1 = logits.cpu().detach().numpy()
    #print(logits_1, logits.shape)
    beam_search_output = ngram_lm_model.decode(logits_1.squeeze(), beam_width=100)
    # beamsearch_prediction, _ = beam_decoder.decode(logits)

    print('argmax prediction    :', argmax_prediction[0])
    print('beamsearch prediction:', beam_search_output)
    
