from infer_with_kenLM.infer_withkenLM import FastConformerWithLM, EvalBeamSearchNGramConfig, load_asr_model
from force_alignment.force_align import AlignmentConfig
import soundfile as sf
from rich import print
import os
from dotenv import load_dotenv
load_dotenv(".env", override=True)

nemo_path = os.environ.get('NEMO_PATH')
kenLM_path = os.environ.get('KENLM_PATH')
asr_device = os.environ.get('ASR_DEVICE')

def infer_asr(audio_path):
    cfg_beam = EvalBeamSearchNGramConfig(kenlm_model_file=kenLM_path)
    cfg_beam.device = asr_device
    forcealign_cfg = AlignmentConfig(batch_size=1)
    forcealign_cfg.device = asr_device
    fast_conformer_model = load_asr_model(nemo_path)
    asr_instance = FastConformerWithLM(beamsearch_cfg=cfg_beam, forcealign_cfg=forcealign_cfg, model=fast_conformer_model)
    audio, _ = sf.read(audio_path)
    dict_response = asr_instance(audio)
    return dict_response
if __name__ == "__main__":
    wav_test = "wavs_test/output_segment_0.wav"
    response = infer_asr(wav_test)
    print(response)
    
    
   
   
