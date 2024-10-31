from transformers.trainer_utils import get_last_checkpoint
# from onnxruntime.quantization.quantize import quantize
from transformers import Wav2Vec2ForCTC
import torch
import os
import argparse
from datetime import datetime


def convert_to_onnx(model_id_or_path, onnx_model_name):
    print(f"Converting {model_id_or_path} to {onnx_model_name}")
    model = Wav2Vec2ForCTC.from_pretrained(model_id_or_path)
    model.eval()
    # print(model)
    audio_len = 16000 * 30

    x = torch.randn(1, audio_len, requires_grad=True)
    # traced_script_module = torch.jit.trace(model, x, strict=False, check_trace=True)
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        onnx_model_name,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input'],  # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={'input': {0: 'batch_size', 1: 'audio_len'}, 'output': {0: 'batch_size'}})  # variable length axes


def quantize_onnx_model(onnx_model_path, quantized_model_path):
    print("Starting quantization...")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QUInt8)

    print(f"Quantized model saved to: {quantized_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        choices=['base', 'large'],
        default='large'
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Whether to use also quantize the model or not",
    )
    args = parser.parse_args()

    # save_dir = f'../output/wav2vec2-{args.model_type}-nguyenvulebinh'
    # last_checkpoint = get_last_checkpoint(save_dir)
    checkpoint_dir = f"/Users/dongnguyentien/Jobs/Projects/Voice/automatic-speech-recognition/ai_modules/speech_to_text/" \
                     f"infer_asr/models/vi/pytorch_models/[large][2023.12.10]new_augm_checkpoint-20960000"
    model_name = f"checkpoint_20960000"
    print(f"last_checkpoint: {os.path.join(checkpoint_dir, model_name)}")

    onnx_model_name = f"wav2vec_{model_name}_{int(datetime.now().timestamp())}.onnx"
    convert_to_onnx(checkpoint_dir, os.path.join(os.path.dirname(__file__), 'converted_onnx', onnx_model_name))

    if (args.quantize):
        quantized_model_name = f"wav2vec_{model_name}_{int(datetime.now().timestamp())}.quant.onnx"
        quantize_onnx_model(onnx_model_name,
                            os.path.join(os.path.dirname(__file__), 'converted_onnx', quantized_model_name))
