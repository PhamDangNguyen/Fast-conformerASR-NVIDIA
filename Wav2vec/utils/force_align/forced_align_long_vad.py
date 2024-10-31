import warnings

warnings.filterwarnings('ignore')
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers.trainer_utils import get_last_checkpoint
import torchaudio
import argparse
import torch
import copy
from forced_align_utils import align
import numpy as np
from pydub import AudioSegment
import contextlib
import webrtcvad
import wave
import collections
import soundfile as sf
from itertools import accumulate
import operator
from math import ceil

ACCEPTED_LENGTH_AUDIO_SEGMENT = 5
MAX_DISTANCE_2_SEGMENT = 3
MIN_LENGTH_AUDIO_SEGMENT = 0.5
MAX_DURATION_AUDIO_SEGMENT = 25
BATCH_SIZE_STT_LARGE_FILE = 4
DEFAULT_SAMPLE_RATE = 16000


class Frame(object):

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class Segment(object):

    def __init__(self, bytes, start, end):
        self.bytes = bytes
        self.start = start
        self.end = end


def format_wave(wave_path):
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
                yield Segment(b''.join([f.bytes for f in voiced_frames]),
                              voiced_frames[0].timestamp,
                              voiced_frames[-1].timestamp + voiced_frames[-1].duration)
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        pass
    if voiced_frames:
        yield Segment(b''.join([f.bytes for f in voiced_frames]),
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


def vad_segment_generator(wavFile, aggressiveness, frame_duration_ms=30, padding_duration_ms=300):
    wavFile = format_wave(wavFile)
    audio, sample_rate, audio_length = read_wave(wavFile)
    vad = webrtcvad.Vad(int(aggressiveness))
    frames = frame_generator(frame_duration_ms, audio, sample_rate)
    segments = vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, list(frames))
    return [segment for segment in segments], sample_rate, audio_length


def segmentation(filename, min_duration=10, max_duration=15, aggressiveness=1):
    vad_segments, sample_rate, audio_length = vad_segment_generator(filename, aggressiveness=aggressiveness)

    output = []
    new_start = None
    for i, s in enumerate(vad_segments):
        start, end = float(s.start), float(s.end)

        if new_start is None:
            new_start = start

        duration = end - new_start

        if duration >= max_duration:
            n_splits = int(duration // max_duration) + 1

            for ni in range(n_splits - 1):
                output.append((new_start + ni * max_duration, new_start + (ni + 1) * max_duration))

            if duration - (n_splits - 1) * max_duration < min_duration:
                output.pop()
                output.append((new_start + (n_splits - 2) * max_duration, new_start + duration))
            else:
                output.append((new_start + (n_splits - 1) * max_duration, new_start + duration))
            new_start = None

        else:
            output.append((new_start, end))
            new_start = None

    if new_start is not None:
        output.append((new_start, end))

    # print(max_duration)
    merge_output = []
    current_seg = None
    for seg in output:
        seg = list(seg)
        if current_seg is None:
            current_seg = seg

        duration = current_seg[1] - current_seg[0]
        # print(duration, current_seg)
        if duration > max_duration:
            merge_output.append(current_seg)
            current_seg = None
        else:
            if seg[-1] - current_seg[0] <= max_duration:
                current_seg[-1] = seg[-1]
            else:
                merge_output.append(current_seg)
                current_seg = seg

    if current_seg is not None:
        merge_output.append(current_seg)

    return merge_output


def load_model(args):
    # load model and tokenizer
    tokenizer = Wav2Vec2CTCTokenizer(
        args.vocab,
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

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    # last_checkpoint = get_last_checkpoint(args.save_folder)
    last_checkpoint = "/home/ndanh/asr-wav2vec/output/checkpoint-17120000_13340000"
    print(f"last_checkpoint: {last_checkpoint}")
    model = Wav2Vec2ForCTC.from_pretrained(last_checkpoint).to(args.device)
    model.eval()
    return processor, model, tokenizer


'''
0.wav: hướng khách hàng là thấu hiểu nhu cầu khách hàng với những chuyên gia kinh doanh và kỹ thuật của cmc việc hiểu khách hàng đòi hỏi phải hiểu sâu về nghiệp vụ chuyên môn mà khách hàng đang làm có đủ năng lực để phân tích nghiệp vụ quy trình trải nghiệm khách hàng thiết kế
1.wav: ra những giải pháp đáp ứng tốt nhất chỉ có đặt mình vào vị trí của khách hàng thì bạn mới hiểu được tâm lý hành vi của họ hiểu được những gì mà họ đang kỳ vọng cảm thông với những sức ép đang đè lên vai khách hàng để có thể hành xử phù hợp mang lại những lợi ích cho khách hàng
'''


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_folder", default="output/wav2vec2-large-nguyenvulebinh-original")
    parser.add_argument("--vocab", default="vocab/vocab_large.json")
    parser.add_argument("--text_file",
                        default="hướng khách hàng là thấu hiểu nhu cầu khách hàng với những chuyên gia kinh doanh và kỹ thuật của cmc việc hiểu khách hàng đòi hỏi phải hiểu sâu về nghiệp vụ chuyên môn mà khách hàng đang làm có đủ năng lực để phân tích nghiệp vụ quy trình trải nghiệm khách hàng thiết kế ra những giải pháp đáp ứng tốt nhất chỉ có đặt mình vào vị trí của khách hàng thì bạn mới hiểu được tâm lý hành vi của họ hiểu được những gì mà họ đang kỳ vọng cảm thông với những sức ép đang đè lên vai khách hàng để có thể hành xử phù hợp mang lại những lợi ích cho khách hàng hướng khách hàng là nắm rõ hành trình khách hàng trong hành trình tiếp cận với cmc khách hàng sẽ trải qua nhiều điểm chạm gặp nhiều nhân viên khác nhau để làm hài lòng khách hàng bạn cần phải hiểu được tâm lý khách hàng tại từng điểm chạm tạo ra những sự thuận tiện và có hành vi ứng xử phù hợp những hành vi này cần được thống nhất và có chung một văn hóa trong suốt hành trình để khách hàng cảm nhận được sự nhất quán hướng khách hàng là mang lại giá trị cho khách hàng khách hàng của cmc chủ yếu là các tổ chức và doanh nghiệp họ sử dụng sản phẩm dịch vụ của cmc để phục vụ khách hàng của họ tốt hơn nâng cao sự hài lòng cho khách hàng của họ để có thể đồng hành cùng với khách hàng thì điều quan trọng nhất là cmc phải mang lại giá trị cho họ nâng tầm uy tín của họ để khách hàng tin tưởng và gắn bó hướng khách hàng là mang lại trải nghiệm trên cả mong đợi cho khách hàng năm hai nghìn không trăm linh sáu trước khi lg ra mắt điện thoại màn hình cảm ứng đầu tiên trên thế giới người dùng di động vẫn nghĩ đơn thuần điện thoại là phải bấm phím nhưng chỉ vài năm sau cảm ứng trở thành xu hướng không thể thay thế được apple và sam sung vận dụng xuất sắc để trở thành hai ông lớn trong ngành di động hiện nay điều khách hàng không nghĩ tới chính là thứ mà lg đã tạo ra nhờ hướng khách hàng trong quá trình làm việc thực tế người cmc cũng đã làm việc với nhiều đối tác khách hàng và mang lại cho họ những trải nghiệm vượt trên mong đợi điều đó đã khiến khách hàng rất bất ngờ và hài lòng")
    parser.add_argument("--filename", default="/home/ndanh/asr-wav2vec/CX_02_031.wav")
    args = parser.parse_args()
    return args


def main(args):
    processor, model, tokenizer = load_model(args)

    filename = args.filename
    speech, sr = sf.read(filename, dtype="float32")

    """
    Initializing dictionary
    """
    vocab_dict = tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab = [x[1] for x in sort_vocab][:-2]
    raw_vocab = vocab
    labels_vocab = copy.deepcopy(raw_vocab)
    labels_vocab[tokenizer.word_delimiter_token_id] = ''
    dictionary = {c: i for i, c in enumerate(labels_vocab)}

    """
    Split long audio into multiple segments
    """
    segments = []
    vad_results = segmentation(filename, min_duration=5, max_duration=15)
    cur_start = 0

    for i, (start, end) in enumerate(vad_results):

        end_cut = int(end * sr)
        if i == 0:
            segments.append(speech[cur_start:end_cut])

            cur_start = end_cut
        elif i == len(vad_results) - 1:
            segments.append(speech[cur_start:])

        else:
            segments.append(speech[cur_start:end_cut])

            cur_start = end_cut

    all_lengths = []
    # Save length of each segment
    for i, segment in enumerate(segments):
        all_lengths.append(segment.shape[0])

    max_lengths = max(all_lengths)
    total_pad_lengths = 0
    pad_each_seg = [0]

    # Calculate padding length, because model ASR auto padding to same size
    for i in range(len(segments)):
        total_pad_lengths += max_lengths - all_lengths[i - 1]

    # Calculate offset of each segment in the order
    for i in range(1, len(segments)):
        # if all_lengths[i-1] < max_lengths:
        offset = max_lengths - all_lengths[i - 1]
        pad_each_seg.append(offset / sr)

    cumulative_pad = list(accumulate(pad_each_seg, operator.add))

    assert len(cumulative_pad) == len(segments)

    attention_masks = torch.zeros((len(segments), max_lengths))
    for i, lengths in enumerate(all_lengths):
        attention_masks[i][:lengths] = 1

    """
    Infer ASR
    """
    input_values = processor(segments, sampling_rate=sr, return_tensors='pt', padding='longest').input_values.to(
        args.device)
    # Simulate in AI Service: config constant of batch size
    BATCH_SIZE = 2
    offset_idx = 0
    tmp_list_batch_logits = []
    with torch.no_grad():
        for i in range(ceil(len(segments) / BATCH_SIZE)):
            logits = model(input_values[
                           offset_idx: offset_idx + BATCH_SIZE if offset_idx + BATCH_SIZE <= len(segments) else len(
                               segments)]).logits
            logits = torch.log_softmax(logits, dim=-1)
            logits = logits.cpu().detach()  # batch x num_window_of_audio
            tmp_list_batch_logits.append(logits)

    # logits = logits.reshape(-1, logits.size(2)).unsqueeze(0) # Shape: [1xnum_windowxembedding_length]

    new_logits = []

    # Thêm token
    for i, tmp_batch_logits in enumerate(tmp_list_batch_logits):
        # tmp_batch_logits: num_batch_list x batch size x num_window x embed size
        for j, tmp_logits in enumerate(tmp_batch_logits):
            # tmp_logits: batch size x num_window x embed size
            if j == len(tmp_batch_logits) - 1 and i == len(tmp_list_batch_logits) - 1:
                new_logits.append(tmp_logits)
            else:
                A = tmp_logits[-2:, 0]  # logit index 0 tương ứng token space của 2 window cuối cùng
                B = tmp_logits[-2:, -1]  # logit index cuối - tương ứng token pad của 2 window cuối cùng
                # tráo đổi 2 logit 2 token này với nhau, để giữa các segment luôn có space ở giữa thay vì pad như hiện tại
                # vì auto logit pad sẽ rất cao và logit của space sẽ rất thấp
                # Upgrade: có thể đo độ dài trung bình các word ở các segment để ko lấy cứng 2 window cuối như hiện tại
                tmp_logits[-2:, 0] = B
                tmp_logits[-2:, -1] = A
                new_logits.append(tmp_logits)

    final_2_logits = torch.cat(new_logits).unsqueeze(0)  # need shape: 1 x num window of full audio x embedding size
    ratio = max_lengths / (tmp_list_batch_logits[0].shape[1])

    """
    Main run alignment
    """
    transcript = args.text_file
    transcript = " " + transcript.replace('\n', ' ').strip() + " "

    list_align, list_space, lengths = [], [], []
    for idx, item in enumerate(align(speech, new_logits[0], transcript, vocab=dictionary,
                                     argmax_prediction=processor.batch_decode(torch.argmax(final_2_logits, dim=-1)[0]),
                                     ratio=ratio, pad_offset=cumulative_pad, pad_lengths=total_pad_lengths,
                                     repeat_limit=2)):
        word, score, start, end = item
        tmp_word = {
            'word': word,
            'start': start,
            'end': end,
            'score': score,
            'length': end - start
        }
        if idx != 0:
            lengths.append(start - list_align[-1]['end'])
            list_space.append({
                'length': start - list_align[-1]['end'],
                'word_before': list_align[-1],
                'word_after': tmp_word
            })

        list_align.append(tmp_word)

    return list_align, list_space


if __name__ == '__main__':
    args = parser_args()

    list_align, list_space = main(args)
    for word in list_align:
        print(f"{word['word']} {word['score']} {word['start']} {word['end']}")
