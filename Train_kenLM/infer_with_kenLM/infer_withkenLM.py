from dataclasses import dataclass, field
from typing import List, Optional
import soundfile as sf
import logging
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
# from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
import torch
from omegaconf import DictConfig, MISSING
import torch
import tempfile
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.submodules import ctc_beam_decoding
from nemo.collections.asr.parts.utils.transcribe_utils import TextProcessingConfig
from nemo.utils import logging
from nemo.collections.asr.models import EncDecCTCModelBPE
from segments import Segment
import json
from tqdm.auto import tqdm
import os


@dataclass
class EvalBeamSearchNGramConfig:
    """
    Evaluate an ASR model with beam search decoding and n-gram KenLM language model.
    """
    # # The path of the '.nemo' file of the ASR model or the name of a pretrained model (ngc / huggingface)
    nemo_model_file: str = MISSING

    # File paths
    input_manifest: str = MISSING  # The manifest file of the evaluation set
    kenlm_model_file: Optional[str] = None  # The path of the KenLM binary model file
    preds_output_folder: Optional[str] = None  # The optional folder where the predictions are stored
    probs_cache_file: Optional[str] = None  # The cache file for storing the logprobs of the model

    # Parameters for inference
    acoustic_batch_size: int = 16  # The batch size to calculate log probabilities
    beam_batch_size: int = 128  # The batch size to be used for beam search decoding
    # device: str = "cpu"  # The device to load the model onto to calculate log probabilities
    device: str = "cuda"
    use_amp: bool = False  # Whether to use AMP if available to calculate log probabilities

    # Beam Search hyperparameters

    # The decoding scheme to be used for evaluation.
    # Can be one of ["greedy", "beamsearch", "beamsearch_ngram"]
    decoding_mode: str = "greedy"

    beam_width: List[int] = field(default_factory=lambda: [32])  # The width or list of the widths for the beam search decoding
    beam_alpha: List[float] = field(default_factory=lambda: [0.7])  # The alpha parameter or list of the alphas for the beam search decoding
    beam_beta: List[float] = field(default_factory=lambda: [1.0])  # The beta parameter or list of the betas for the beam search decoding

    # Can be one of ["flashlight", "pyctcdecode", "beam"]
    decoding_strategy: str = "flashlight"
    decoding: ctc_beam_decoding.BeamCTCInferConfig = field(default_factory=lambda: ctc_beam_decoding.BeamCTCInferConfig(beam_size=128))
    
    text_processing: Optional[TextProcessingConfig] = field(default_factory=lambda: TextProcessingConfig(
        punctuation_marks = "",
        separate_punctuation = False,
        do_lowercase = False,
        rm_punctuation = False,
    ))

class FastConformerASR(EncDecCTCModelBPE):

    def __init__(self, cfg: DictConfig, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)
    
    @torch.no_grad()
    def transcribe(
        self,
        paths2audio_files: List[str],
        batch_size: int = 4,
        logprobs: bool = False,
        return_hypotheses: bool = False,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
    ) -> List[str]:
        """
        If modify this function, please remember update transcribe_partial_audio() in
        nemo/collections/asr/parts/utils/trancribe_utils.py

        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:
            paths2audio_files: (a list) of paths to audio files. \
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            logprobs: (bool) pass True to get log probabilities instead of transcripts.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
        Returns:
            A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2audio_files
        """
        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}

        if return_hypotheses and logprobs:
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` can be True at any given time."
                "Returned hypotheses will contain the logprobs."
            )

        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)
        # We will store transcriptions here
        hypotheses = []
        all_hypotheses = []
        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device
        dither_value = self.preprocessor.featurizer.dither
        pad_to_value = self.preprocessor.featurizer.pad_to

        try:
            self.preprocessor.featurizer.dither = 0.0
            self.preprocessor.featurizer.pad_to = 0
            # Switch model to evaluation mode
            self.eval()
            # Freeze the encoder and decoder modules
            self.encoder.freeze()
            self.decoder.freeze()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
                    for audio_file in paths2audio_files:
                        entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                        fp.write(json.dumps(entry) + '\n')

                config = {
                    'paths2audio_files': paths2audio_files,
                    'batch_size': batch_size,
                    'temp_dir': tmpdir,
                    'num_workers': num_workers,
                    'channel_selector': channel_selector,
                }

                if augmentor:
                    config['augmentor'] = augmentor

                temporary_datalayer = self._setup_transcribe_dataloader(config)
                for test_batch in tqdm(temporary_datalayer, desc="Transcribing", disable=not verbose):
                    logits, logits_len, greedy_predictions = self.forward(
                        input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                    )
                    probs = []
                    for idx in range(logits.shape[0]):
                        lg = logits[idx][: logits_len[idx]]
                        probs.append(lg.cpu().numpy())
                    else:
                        current_hypotheses, all_hyp = self.decoding.ctc_decoder_predictions_tensor(
                            logits, decoder_lengths=logits_len, return_hypotheses=return_hypotheses,
                        )
                        logits = logits.cpu()
                
                        if return_hypotheses:
                            # dump log probs per file
                            for idx in range(logits.shape[0]):
                                current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]
                                if current_hypotheses[idx].alignments is None:
                                    current_hypotheses[idx].alignments = current_hypotheses[idx].y_sequence
                        if all_hyp is None:
                            hypotheses += current_hypotheses
                        else:
                            hypotheses += all_hyp
                    
                    del greedy_predictions
                    del logits
                    del test_batch
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            self.preprocessor.featurizer.dither = dither_value
            self.preprocessor.featurizer.pad_to = pad_to_value
            if mode is True:
                self.encoder.unfreeze()
                self.decoder.unfreeze()
            logging.set_verbosity(logging_level)
        # print(hypotheses)
        return hypotheses,probs

class FastConformerWithLM:

    def __init__(
        self, cfg: EvalBeamSearchNGramConfig,
        beam_width=32, beam_alpha=1.0, beam_beta=1.0,
        return_best_hypothesis=False,
    ):  
        if cfg.nemo_model_file.endswith(".nemo"):
            model = FastConformerASR.restore_from(
                cfg.nemo_model_file, map_location=torch.device(cfg.device)
            )
        else:
            model = nemo_asr.models.EncDecCTCModelBPE.load_from_checkpoint(
                cfg.nemo_model_file, map_location=torch.device(cfg.device)
            )
        cfg.decoding.beam_size = beam_width
        cfg.decoding.beam_alpha = beam_alpha
        cfg.decoding.beam_beta = beam_beta
        cfg.decoding.return_best_hypothesis = return_best_hypothesis
        cfg.decoding.kenlm_path = cfg.kenlm_model_file
        cfg.decoding.preserve_word_confidence = True
        self.cfg = cfg
        self.asr_model = model
    
    def _disable_logging(self):
        logging.set_verbosity(logging.CRITICAL)
    
    def _enable_logging(self):
        logging.set_verbosity(logging.INFO)
        
    def _enable_preserve_alignments(self):
        self.cfg.decoding_mode = "greedy"
        self.cfg.decoding_strategy = "greedy"
        decoding_cfg = self.asr_model.cfg.decoding
        decoding_cfg.preserve_alignments = True
        decoding_cfg.compute_timestamps = True
        decoding_cfg.strategy = self.cfg.decoding_strategy
        decoding_cfg.beam = self.cfg.decoding
        decoding_cfg.confidence_cfg.preserve_word_confidence = True
        decoding_cfg.confidence_cfg.preserve_token_confidence = True
        decoding_cfg.confidence_cfg.preserve_frame_confidence = True
        decoding_cfg.confidence_cfg.method_cfg.name = "max_prob"
        # decoding_cfg.confidence_cfg.measure_cfg.name = "max_prob"
        # print(decoding_cfg.method_cfg.name)
        # print(decoding_cfg)
        self._disable_logging()
        self.asr_model.change_decoding_strategy(decoding_cfg)
        self._enable_logging()

    def _enable_beamsearch(self):
        self.cfg.decoding_mode = "beamsearch_ngram"
        self.cfg.decoding_strategy = "beam"
        decoding_cfg = self.asr_model.cfg.decoding
        decoding_cfg.preserve_alignments = False
        decoding_cfg.compute_timestamps = False
        decoding_cfg.confidence_cfg.preserve_word_confidence = False
        decoding_cfg.confidence_cfg.preserve_token_confidence = False
        decoding_cfg.confidence_cfg.preserve_frame_confidence = False
        decoding_cfg.strategy = self.cfg.decoding_strategy
        decoding_cfg.beam = self.cfg.decoding
        self._disable_logging()
        # print(decoding_cfg)
        self.asr_model.change_decoding_strategy(decoding_cfg)
        self._enable_logging()
        
    def transcribe(self, audio_file: str,batch_size=3):
        self._enable_preserve_alignments()
        outputs,_= self.asr_model.transcribe(audio_file,return_hypotheses=True,batch_size=batch_size)
        # print(outputs)
        outputs_text = []
        output_timesteps = []
        for index,output in enumerate(outputs):
            # print("---------------------------------------------------------------------")
            timesteps, _, probs_batch = outputs[index].timestep,outputs[index].timestep,[outputs[index].y_sequence]
            print(len(timesteps["word"]))
            print(len(output.word_confidence))

            if len(timesteps["word"]) == len(output.word_confidence):
                raw_segments = [Segment(w["word"], w["start_offset"], w["end_offset"],output.word_confidence[i]) for i, w in enumerate(timesteps["word"])]
            else:
                raw_segments = []
                for i,w in enumerate(timesteps["word"]):
                    try: 
                        raw_segments.append(Segment(w["word"], w["start_offset"], w["end_offset"], output.word_confidence[i]))
                    except:
                        raw_segments.append(Segment(w["word"], w["start_offset"], w["end_offset"], output.word_confidence[0]))

            raw_segments = [seg for seg in raw_segments if seg.word != ""]
            self._enable_beamsearch()
            probs_lens = torch.tensor([prob.shape[0] for prob in probs_batch])
            with torch.no_grad():
                packed_batch = torch.zeros(len(probs_batch), max(probs_lens), probs_batch[0].shape[-1], device='cpu')

                for prob_index in range(len(probs_batch)):
                    packed_batch[prob_index, : probs_lens[prob_index], :] = torch.tensor(
                        probs_batch[prob_index], device=packed_batch.device, dtype=packed_batch.dtype
                    )

                _, beams_batch = self.asr_model.decoding.ctc_decoder_predictions_tensor(
                    packed_batch, decoder_lengths=probs_lens, return_hypotheses=True,
                )
            kenlm_text = beams_batch[0][0].text
            out_timestep = []
            for i,timestep in enumerate(timesteps["word"]):
                try: 
                    out_timestep.append((timestep["start_offset"],timestep["end_offset"],output.word_confidence[i]))
                except:
                    out_timestep.append((timestep["start_offset"],timestep["end_offset"],output.word_confidence[0]))

            output_timesteps.append(out_timestep)
            outputs_text.append([raw_segments,kenlm_text])
          
        return outputs_text,output_timesteps    

    def get_confidence(self,audio_file):
        self._enable_preserve_alignments()
        # timesteps, _, logits = self.asr_model.transcribe(audio_file)
        out_put,probs = self.asr_model.transcribe(audio_file,return_hypotheses=True)
        mean_confidence = sum(out_put[0].word_confidence)/len(out_put[0].word_confidence)
        return mean_confidence

        

def get_duration(audio_path):
    audio, sr = sf.read(audio_path)
    return len(audio) / sr

from rich import print
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Infer with KenLM')
    parser.add_argument("--nemo_model", required=True, default=None, type=str, help='ASR model')
    parser.add_argument("--kenlm_model", required=True, default=None, type=str, help='LM model')
    args = parser.parse_args()
    
    cfg = EvalBeamSearchNGramConfig(beam_alpha=0.7, beam_beta=1.0, beam_width=32)
    cfg.nemo_model_file  = args.nemo_model
    cfg.kenlm_model_file = args.kenlm_model

    fast_conformer = FastConformerWithLM(cfg=cfg, beam_alpha=0.5, beam_beta=2.0, beam_width=64) 
    check,timesteps = fast_conformer.transcribe(["/mnt/driver/pdnguyen/data_record/Audio_boss_chinh/chunk_4.wav"],batch_size=1)
    print(check)

    # root_dir = "/home/pdnguyen/Audio_Augmentation/fubon_11_10_2024/[Lỗi 11.10] File 4"
    # from pathlib import Path

    # wav_files = Path(root_dir).glob('**/*.wav')
    # wav_files = [str(file) for file in wav_files]
    # wav_files = sorted(wav_files, key=lambda x: float(x.split("/")[-1].split(".")[0]))
    # # with open(f'{root_dir}/metadata.csv', 'w', encoding='utf8') as fp:
    # #     print(f'file,text,duration',file=fp)
    # #     for file_path in wav_files:
    # #         check,timesteps = fast_conformer.transcribe([file_path],batch_size=1)
    # #         text = check[0][1]
    # #         duration = get_duration(file_path)
    # #         print(f'{file_path},{str(text)},{duration}',file=fp)
    # # print("done")

    # for file_path in wav_files:
    #     check,timesteps = fast_conformer.transcribe([file_path],batch_size=1)
    #     print(file_path)
    #     print(check[0][1])
    