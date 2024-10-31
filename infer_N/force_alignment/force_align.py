import sys
from dataclasses import dataclass, is_dataclass
from typing import List, Optional
from force_alignment.utils.data_prep import Word
import torch
from omegaconf import OmegaConf
from force_alignment.utils.data_prep import (
    add_t_start_end_to_utt_obj,
    get_batch_starts_ends,
    get_batch_variables,
    get_manifest_lines_batch,
)
from force_alignment.utils.viterbi_decoding import viterbi_decoding
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
from nemo.utils import logging

from typing import List, Optional
from omegaconf import MISSING

@dataclass
class AlignmentConfig:
    # Required configs
    pretrained_name: Optional[str] = None
    model_path: Optional[str] = None
    output_dir: Optional[str] = None
    device : str = "cuda"
    # General configs
    align_using_pred_text: bool = False
    transcribe_device: Optional[str] = None
    viterbi_device: Optional[str] = None
    batch_size: int = 1
    use_local_attention: bool = True
    additional_segment_grouping_separator: Optional[str] = None
    audio_filepath_parts_in_utt_id: int = 1

    # Buffered chunked streaming configs
    use_buffered_chunked_streaming: bool = False
    chunk_len_in_secs: float = 1.6
    total_buffer_in_secs: float = 4.0
    chunk_batch_size: int = 32

    # Cache aware streaming configs
    simulate_cache_aware_streaming: Optional[bool] = False


class InnferConformer(FrameBatchASR):
    def __init__(self, asr_model, frame_len=1.6, total_buffer=4.0, batch_size=4):
        super().__init__(asr_model, frame_len, total_buffer, batch_size)
        self.additional_param = None

    def transcribe(self, tokens_per_chunk: int, delay: int, keep_logits=False):
        self.infer_logits(keep_logits)
        self.unmerged = []
        for pred in self.all_preds:
            decoded = pred.tolist()
            self.unmerged += decoded[len(decoded) - 1 - delay : len(decoded) - 1 - delay + tokens_per_chunk]
        hypothesis = self.greedy_merge(self.unmerged)
        if not keep_logits:
            return hypothesis

        all_logits = []
        for log_prob in self.all_logits:
            T = log_prob.shape[0]
            log_prob = log_prob[T - 1 - delay : T - 1 - delay + tokens_per_chunk, :]
            all_logits.append(log_prob)
        all_logits = torch.concat(all_logits, 0)
        return hypothesis, all_logits

def map_value(x, min_old, max_old, min_new, max_new):
    """
    Map a value from one range to another range
    Args:
        x: value to be mapped
        min_old: minimum of the old range
        max_old: maximum of the old range
        min_new: minimum of the new range
        max_new: maximum of the new range
    Returns:
        The mapped value x in new range
    """
    a = (max_new - min_new) / (max_old - min_old)
    b = min_new - a * min_old
    return a * x + b

def calculate_confidence(timestep_raw, alig_timestep):
    """
    Args:
        timestep_raw: list of (start, end, confidence) tuples
        alig_timestep: list of (start, end, [words.t_start, words.t_end, words.text]) tuples
    Returns:
        list of (word, confidence) tuples
    """
    results = []
    for start_b, end_b, word_info in alig_timestep:
        relevant_confidences = []
        for item in timestep_raw:
            if item[1] >= end_b and item[0] <= start_b:
                relevant_confidences.append(item[2])
            elif (start_b < item[1] and end_b > item[1]) or (start_b < item[0] and end_b > item[0]):
                relevant_confidences.append(item[2])
            elif (start_b < item[0] and end_b > item[1]):
                relevant_confidences.append(item[2])
            
        if relevant_confidences:
            avg_confidence = sum(relevant_confidences) / len(relevant_confidences)
        else:
            avg_confidence = 0.99999999999999999999988888888888888888888
        results.append((word_info[2], avg_confidence))
    return results

def run_alignment(cfg: AlignmentConfig, audio_meta, model, timesteps_raw):
    """
    This function will call the alignment model to get the alignment for each utterance in the batch.
    Args:
        cfg: config object
        audio_meta: list of audio files
        model: ASR model object
        timesteps_raw: list of (start, end, confidence) tuples
    Returns:
        list of (word, confidence) tuples
    """

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)
    
    # # init devices
    transcribe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if cfg.transcribe_device is None else torch.device(cfg.transcribe_device)
    logging.info(f"Device to be used for transcription step (`transcribe_device`) is {transcribe_device}")

    viterbi_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    if model is not None:
        logging.info("Load model fast Cfm")
        model.eval()
    if isinstance(model, EncDecHybridRNNTCTCModel):
        model.change_decoding_strategy(decoder_type="ctc")

    if cfg.use_local_attention:
        logging.info(
            "Flag use_local_attention is set to True => will try to use local attention for model if it allows it"
        )
        model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[64, 64])

    buffered_chunk_params = {}
    starts, ends = get_batch_starts_ends(manifest_filepath=audio_meta, batch_size=cfg.batch_size)
    output_timestep_duration = None

    # # get alignment and save in CTM batch-by-batch
    for start, end in zip(starts, ends):
        manifest_lines_batch = get_manifest_lines_batch(audio_meta, start, end)
        (log_probs_batch, y_batch, T_batch, U_batch, utt_obj_batch, output_timestep_duration,) = get_batch_variables(
            manifest_lines_batch,
            model,
            cfg.additional_segment_grouping_separator,
            cfg.align_using_pred_text,
            cfg.audio_filepath_parts_in_utt_id,
            output_timestep_duration,
            cfg.simulate_cache_aware_streaming,
            cfg.use_buffered_chunked_streaming,
            buffered_chunk_params,
        )
        alignments_batch = viterbi_decoding(log_probs_batch, y_batch, T_batch, U_batch, viterbi_device)
        for utt_obj, alignment_utt in zip(utt_obj_batch, alignments_batch):
            utt_obj = add_t_start_end_to_utt_obj(utt_obj, alignment_utt, output_timestep_duration)
            arr_word = []
            arr_start = []
            arr_end = []
            for words in utt_obj.segments_and_tokens[1].words_and_tokens:
                if type(words) is Word:
                    if(words.text != ""):
                        arr_word.append([words.t_start, words.t_end, words.text])
                        arr_start.append(words.t_start)
                        arr_end.append(words.t_end)
            # dev infer single signal
            min_old = min(arr_start)
            max_old = max(arr_end)
            arr_get_start_raw = [x[0] for x in timesteps_raw]
            arr_get_end_raw = [x[1] for x in timesteps_raw]
            
            min_new = min(arr_get_start_raw) #get max value in arr_get_start_raw
            max_new = max(arr_get_end_raw) #get min value in arr_get_end_raw

            arr_time_step = [(int(map_value(arr_start[i], min_old, max_old, min_new, max_new)), int(map_value(arr_end[i], min_old, max_old, min_new, max_new)), arr_word[i]) for i,x in enumerate(arr_start)]
            #get confidence
            rs_confidence = calculate_confidence(timesteps_raw,arr_time_step)
        return arr_word,rs_confidence