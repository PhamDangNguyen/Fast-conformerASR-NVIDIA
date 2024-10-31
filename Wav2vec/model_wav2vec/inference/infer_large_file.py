from pydub import AudioSegment
import collections
import contextlib
import webrtcvad
import wave
import os
import math
import numpy as np
import soundfile as sf
from transformers.trainer_utils import get_last_checkpoint
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
import torch
import torch.nn as nn
import kenlm
import copy
from statistics import mean

ACCEPTED_LENGTH_AUDIO_SEGMENT = 12
MAX_DISTANCE_2_SEGMENT = 3
MIN_LENGTH_AUDIO_SEGMENT = 0.5
MAX_DURATION_AUDIO_SEGMENT = 25
BATCH_SIZE_STT_LARGE_FILE = 4
DEFAULT_SAMPLE_RATE = 16000
TIME_STEP_WINDOW: float = 0.02
MIN_PROB_WORD = 0.4

class Frame(object):

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class VadSegment(object):

    def __init__(self, bytes, start, end):
        self.bytes = bytes
        self.start = start
        self.end = end


def format_wave(wave_path):
    print(f"wave_path: {wave_path}")
    if wave_path.endswith(('mp3', 'MP3')):
        sound = AudioSegment.from_mp3(wave_path)
        wave_path = wave_path[:-4] + '.wav'
        sound = sound.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        sound.export(wave_path, format='wav')
    elif wave_path.endswith(('wav', 'WAV')):
        sound = AudioSegment.from_wav(wave_path)

    if sound.channels > 1 and sound.frame_rate != 16000 and sound.sample_width != 2:
        sound = sound.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        sound.export(wave_path, format='wav')

    return wave_path


def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000)
        frames = wf.getnframes()
        pcm_data = wf.readframes(frames)
        duration = frames / sample_rate
        return pcm_data, sample_rate, duration


def write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield VadSegment([f.bytes for f in voiced_frames],
                                 voiced_frames[0].timestamp,
                                 voiced_frames[-1].timestamp + voiced_frames[-1].duration)
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        pass
    if voiced_frames:
        yield VadSegment([f.bytes for f in voiced_frames],
                         voiced_frames[0].timestamp,
                         voiced_frames[-1].timestamp + voiced_frames[-1].duration)


def frame_generator(frame_duration_ms, audio, sample_rate):
    frame_duration_s = frame_duration_ms / 1000.0
    frame_byte_count = int(sample_rate * frame_duration_s * 2)
    offset = 0
    timestamp = 0.0
    while offset + frame_byte_count - 1 < len(audio):
        yield Frame(audio[offset:offset + frame_byte_count], timestamp, frame_duration_s)
        timestamp += frame_duration_s
        offset += frame_byte_count


def vad_segment_generator(audio_data, sample_rate, aggressiveness,
                          frame_duration_ms=20, padding_duration_ms=200):
    vad = webrtcvad.Vad(int(aggressiveness))
    frames = frame_generator(frame_duration_ms, audio_data, sample_rate)
    segments = vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, list(frames))
    return [segment for segment in segments]


class NormalizeSegment(object):
    """
    Normalize segment can be combined from other segments to maximize length and minimize data loss due to vad
    """
    audio_path: str

    def __init__(self, start, end, speech, length):
        self.start = start # Second
        self.end = end # Second
        self.speech = speech
        self.length = length  # Độ dài này là độ dài thật của audio này đã cắt hết toàn bộ phần ko có tiếng nói, do đó
        # không thể lấy start - end để lấy length được


def segment_large_file_ver_2(wav_name, audio,aggressiveness=1):
    """
    Sử dụng vad trên int 16bit vì vad hoạt động trên bytes, độ dài item là 2 bytes,
     nếu qua float 32 sẽ dùng 4 bytes, chuỗi bytes sẽ dài ra 2 lần
    :param filename:
    :param tmp_dir:
    :param aggressiveness: độ quyết liệt, giá trị càng lớn thì sẽ cắt càng chi tiết, có thể lọt nhiễu tiếng nói
    :param audio: instance of Audio class
    :return:
    """
    
    if wav_name is not None:
        # wavFile = format_wave(filename)
        # audio_data, sample_rate, audio_length = read_wave(wav_name)
        audio_data, sample_rate = sf.read(wav_name, dtype='int16')
        # audio_data_int16 = (audio_data_float32 * np.iinfo(np.int16).max).astype(np.int16)
        audio_data = audio_data.tobytes()
    else:
        if audio.dtype == np.float32:
            audio_data_int16 = (audio * np.iinfo(np.int16).max).astype(np.int16)
            audio_data = audio_data_int16.tobytes()
        else:
            # Int16
            audio_data = audio.tobytes()
        sample_rate = DEFAULT_SAMPLE_RATE
    # print(f"audio vad length: {audio.duration}, sample rate: {sample_rate}, audio type: {audio_data_int16.dtype}, {audio.speech.dtype}")
    vad_segments = vad_segment_generator(audio_data, sample_rate, aggressiveness=aggressiveness)
    output = []  # [(start, end, audio_path)]
    count = 1
    tmp_norm_seg = None  # NormalizeSegment
    # print("#######")
    for i, s in enumerate(vad_segments):
        # Lỗi ở đây là start, end thì theo audio gốc và start cái sau không liền end cái trước (khoảng trống ở giữa thì mất dữ liệu),
        # Trong khi đó, vòng for chạy qua nhiều segment thì tính duration lại lấy end segment hiện tại từ start segment trước
        # khiến thực tế tính độ dài segment không chính xác, đồng thời gây mất mát dữ liệu do vad bỏ những đoạn rất ngắn nhưng
        # ảnh hưởng trực tiếp đến audio vì bị mất âm các từ
        # Chỉnh sửa: thêm 1 yếu tố nữa là khoảng cách giữa các segment, nếu >5s thì lấy s.bytes, còn k thì merge cơ học, k bỏ các phần audio mà vad coi là k có âm thanh
        start, end = float(s.start), float(s.end)
        # print(f"vad result: {start} - {end}")
        duration = end - start

        if tmp_norm_seg:
            distance = start - tmp_norm_seg.end
            if distance >= MAX_DISTANCE_2_SEGMENT:
                # write_wave(tmp_dir + f'/{count}.wav', b''.join([f for f in tmp_norm_seg.speech]), sample_rate)
                if tmp_norm_seg.length >= MIN_LENGTH_AUDIO_SEGMENT:
                    # await run_sync_function_in_async(sf.write, tmp_dir + f'/{int(datetime.now().timestamp())}_{count}.wav', tmp_norm_seg.speech, sample_rate)
                    # output.append((tmp_norm_seg.start, tmp_norm_seg.end, tmp_dir + f'/{count}.wav'))
                    output.append((tmp_norm_seg.start, tmp_norm_seg.end))
                    count += 1

                tmp_norm_seg = NormalizeSegment(start=start, end=end,
                                                speech=audio[int(start * sample_rate):int(end * sample_rate)],
                                                length=duration)
            else:
                tmp_norm_seg.end = end
                tmp_norm_seg.length += duration
                tmp_norm_seg.speech = audio[int(tmp_norm_seg.start * sample_rate):int(tmp_norm_seg.end * sample_rate)]

        else:
            tmp_norm_seg = NormalizeSegment(start=start, end=end,
                                            speech=audio[int(start * sample_rate):int(end * sample_rate)],
                                            length=duration)

        if tmp_norm_seg.length > MAX_DURATION_AUDIO_SEGMENT:
            n_splits = int(tmp_norm_seg.length // MAX_DURATION_AUDIO_SEGMENT) + 1
            split_data = math.floor(len(tmp_norm_seg.speech) / n_splits)
            split_val = tmp_norm_seg.length / n_splits
            cur_split_data = 0
            for ni in range(n_splits):
                if ni == n_splits - 1:
                    # await run_sync_function_in_async(sf.write, tmp_dir + f'/{count}.wav',
                    #                                  tmp_norm_seg.speech[cur_split_data:],
                    #                                  sample_rate)
                    output.append(
                        (tmp_norm_seg.start + ni * split_val, tmp_norm_seg.end))
                else:
                    next_split_data = cur_split_data + split_data
                    # await run_sync_function_in_async(sf.write, tmp_dir + f'/{count}.wav',
                    #                                  tmp_norm_seg.speech[cur_split_data:next_split_data],
                    #                                  sample_rate)
                    output.append((tmp_norm_seg.start + ni * split_val, tmp_norm_seg.start + (ni + 1) * split_val))
                    cur_split_data = next_split_data
                count += 1

            tmp_norm_seg = None

        elif tmp_norm_seg.length >= ACCEPTED_LENGTH_AUDIO_SEGMENT or i == len(vad_segments) - 1:
            # write_wave(tmp_dir + f'/{count}.wav', b''.join([f for f in tmp_norm_seg.speech]), sample_rate)
            # await run_sync_function_in_async(sf.write, tmp_dir + f'/{count}.wav', tmp_norm_seg.speech, sample_rate)
            output.append((tmp_norm_seg.start, tmp_norm_seg.end))
            count += 1
            tmp_norm_seg = None

    return output, sample_rate


def create_batch_long_audio(wav_name, batch_size=BATCH_SIZE_STT_LARGE_FILE):
    audio, sr = sf.read(wav_name, dtype="float32")
    output_vad, sample_rate = segment_large_file_ver_2(wav_name=wav_name,audio=audio)
    Segment = []
    for i, (start, end) in enumerate(output_vad):
        Segment.append(audio[int(start * sample_rate):int(end * sample_rate)])

    return Segment

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
        
        return logits

class Preprocessor(nn.Module):

    def __init__(self, last_checkpoint,model_type='large', device='cpu'):
        super().__init__()
        # load model and tokenizer
        tokenizer = Wav2Vec2CTCTokenizer(
            f'../vocab/vocab_large.json',
            unk_token='<unk>', 
            pad_token='<pad>', 
            word_delimiter_token=' '
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
    
def load_all_model(last_checkpoint):
    device = "cpu"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pytorch_model = CMCWav2vec(last_checkpoint, 'large', device=device)
    return pytorch_model,device

# beam search
def get_decoder_ngram_model(tokenizer, ngram_lm_path):
    vocab_dict = tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab = [x[1] for x in sort_vocab][:-1]
    vocab_list = vocab
    # print(len(vocab_list), tokenizer.unk_token_id, tokenizer.word_delimiter_token_id)
    # vocab_list[tokenizer.unk_token_id] = ""
    # vocab_list[tokenizer.word_delimiter_token_id] = " "
    alphabet = Alphabet.build_alphabet(vocab_list)
    lm_model = kenlm.Model(ngram_lm_path)
    decoder = BeamSearchDecoderCTC(alphabet, language_model=LanguageModel(lm_model))
    return decoder

def infer(audio,pytorch_model,ngram_lm_model):
    # logits = pytorch_model(audio)[0]

    # argmax_prediction = pytorch_model.processor.decode(logits)
    # beam_search_output = ngram_lm_model.decode(logits.detach().cpu().numpy().squeeze(), beam_width=500)

    logits = pytorch_model(audio).detach().cpu()
    logits_tensor = copy.deepcopy(logits.to(torch.float32))
    argmax_prediction = pytorch_model.processor.processor.batch_decode(torch.argmax(logits, dim=-1)[0])
    beam_search_output = ngram_lm_model.decode_beams(logits.numpy().squeeze(), beam_width=500)[0]
    return argmax_prediction, beam_search_output, torch.log_softmax(logits, dim=-1), logits_tensor

if __name__ == '__main__':
    Segments = create_batch_long_audio(wav_name="/home/ndanh/asr-wav2vec/CX_02_031.wav")
    results = []
    save_dir = f'output/wav2vec2-large-nguyenvulebinh-original/'
    #last_checkpoint = get_last_checkpoint(save_dir)
    last_checkpoint =  '/home/ndanh/asr-wav2vec/output/checkpoint-17120000_13340000'
    # last_checkpoint = '/home/ndanh/asr-wav2vec/output/checkpoint-17120000'
    lm_path = "ngram_lm/[2022_05]vi_4gram.binary"
    
    print(f"last_checkpoint: {last_checkpoint}")
    pytorch_model,device = load_all_model(last_checkpoint)
    ngram_lm_model = get_decoder_ngram_model(pytorch_model.processor.processor.tokenizer, lm_path)
    for segment in Segments:
        duration = segment.shape[0] / 16000
        p, p_lm, logits, logits_tensor = infer(segment,pytorch_model,ngram_lm_model)
        softmax_pred = torch.softmax(logits_tensor, dim=-1)
        max_elements, _ = torch.max(softmax_pred, dim=-1)
        max_elements = max_elements.tolist()
        end_in_time_step_window = math.floor(duration / TIME_STEP_WINDOW)

        all_words = []
        for word_index, (word, (start, end)) in enumerate(p_lm[2]):
            # normalize end, something end word larger than segment duration
            word = word.lower()
            end = end if end <= end_in_time_step_window else end_in_time_step_window

            confident_score_list = max_elements[0][start:end]
            
            confident_score = mean(confident_score_list) if len(confident_score_list) > 0 else 0.0
            if confident_score <= MIN_PROB_WORD:
                
                continue
            
            all_words.append(word)
        pred_trans = " ".join(word for word in all_words)
        pred_trans = pred_trans.strip()
        results.append(pred_trans)

    final_result = " ".join(result for result in results)
    print(final_result)