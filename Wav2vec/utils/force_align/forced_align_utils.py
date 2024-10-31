import torch
import torchaudio
from dataclasses import dataclass

import matplotlib.pyplot as plt
ACCEPTED_LENGTH_AUDIO_SEGMENT = 5


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float
    token: str
    is_sep_or_pad: bool
    time_offset: float


# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float
    time_offset: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # trellis = torch.empty((num_frame + 1, num_tokens + 1))
    # trellis[0, 0] = 0
    # trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    # trellis[0, -num_tokens:] = -float("inf")
    # trellis[-num_tokens:, 0] = float("inf")

    trellis = torch.zeros((num_frame, num_tokens))

    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1:, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            # trellis[t, :-1] + emission[t, tokens],
            trellis[t, :-1] + emission[t, tokens[1:]],
        )

    return trellis


def backtrack(trellis, emission, tokens, argmax_prediction, pad_offset=None, blank_id=0):  ## blank_id la <pad> token
    if pad_offset is not None:
        num_segments = len(pad_offset)
    else:
        num_segments = 1

    n_frame = trellis.size(0) / num_segments

    t, j = trellis.size(0) - 1, trellis.size(1) - 1  # t: time_step, jL token_index

    if pad_offset is not None:
        path = [Point(j, t, emission[t, blank_id].exp().item(), argmax_prediction[t],
                      True if argmax_prediction[t] in ['<pad>', ''] else False, pad_offset[int(t / n_frame)])]
    else:
        path = [Point(j, t, emission[t, blank_id].exp().item(), argmax_prediction[t],
                      True if argmax_prediction[t] in ['<pad>', ''] else False, 0)]
    while j > 0:
        # Should not happen but just in case
        assert t > 0

        # 1. Figure out if the current position was stay or change
        # Frame-wise score of stay vs change
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1

        # Store the path with frame-wise probability.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        # a = argmax_prediction[j] in ['<pad>', '']
        if pad_offset is None:
            path.append(
                Point(j, t, prob, argmax_prediction[t], True if argmax_prediction[t] in ['<pad>', ''] else False, 0))
        else:
            path.append(
                Point(j, t, prob, argmax_prediction[t], True if argmax_prediction[t] in ['<pad>', ''] else False,
                      pad_offset[int(t / n_frame)]))

        if changed > stayed:
            j -= 1
    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        if pad_offset is not None:
            path.append(
                Point(j, t, prob, argmax_prediction[t], True if argmax_prediction[t] in ['<pad>', ''] else False,
                      pad_offset[int(t / n_frame)]))
        else:
            path.append(Point(j, t - 1, prob, argmax_prediction[t],
                              False if argmax_prediction[t] not in ['<pad>', ''] else True, 0))
        t -= 1

    return path[::-1]


def merge_repeats(path, transcript, repeat_limit=-1):
    assert repeat_limit == -1 or repeat_limit > 0
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = []
        for k in range(i1, i2):
            if not path[k].is_sep_or_pad:
                score.append(path[k].score)
        # score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        if len(score) == 0:
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        else:
            score = sum(score) / len(score)

        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1 if repeat_limit == -1 else min(path[i1].time_index + repeat_limit + 1,
                                                                           path[i2 - 1].time_index + 1),
                score,
                path[i1].time_offset
            )
        )
        i1 = i2
    return segments


# Merge words
def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score, segments[i1].time_offset))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


def align(speech, logits, transcript, argmax_prediction, vocab, pad_offset=None, pad_lengths=0, ratio=1.0,
          repeat_limit=-1, separator=" ", blank_id=-1):
    tokens = [vocab[c] if c != separator else vocab[''] for c in transcript]
    trellis = get_trellis(logits, tokens, blank_id=blank_id)

    path = backtrack(trellis, logits, tokens, argmax_prediction, pad_offset=pad_offset, blank_id=blank_id)

    segments = merge_repeats(path, transcript, repeat_limit=repeat_limit)

    word_segments = merge_words(segments, separator=separator)

    # ratio = speech.shape[0] / (trellis.shape[0] - 1)
    # ratio = (speech.shape[0] + pad_lengths) / (trellis.shape[0] - 1)

    for word in word_segments:
        start = int(ratio * word.start)
        end = int(ratio * word.end)
        # yield (speech[start:end], word.label, word.score, start, end)
        yield word.label, word.score, start / 16000 - word.time_offset, end / 16000 - word.time_offset


if __name__ == '__main__':

    torch.random.manual_seed(0)

    SPEECH_FILE = torchaudio.utils.download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
    device = torch.device('cpu')

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    with torch.inference_mode():
        speech, fs = torchaudio.load(SPEECH_FILE)
        logits, _ = model(speech.to(device))
        logits = torch.log_softmax(logits, dim=-1)

    logits = logits[0].cpu().detach()
    # print(emission.shape)

    text = "I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT"
    vocab = {c: i for i, c in enumerate(labels)}

    for item in align(speech, logits, text, vocab):
        print(item[1:])
        segment, word, score, start, end = item
        torchaudio.save(f'rac/{word}.wav', segment, fs)
