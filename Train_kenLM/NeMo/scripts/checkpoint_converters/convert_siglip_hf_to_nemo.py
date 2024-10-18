# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Requires HF transformers updated to support Siglip Models
    python /opt/NeMo/scripts/checkpoint_converters/convert_siglip_hf_to_nemo.py \
      --input_name_or_path=google/siglip-so400m-patch14-384 \
      --output_path=test.nemo
"""

import os
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf
from transformers import AutoModel, AutoProcessor

from nemo.collections.multimodal.models.vision_language_foundation.clip.megatron_clip_models import MegatronCLIPModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils import logging


def create_rename_keys(num_hidden_layers):
    rename_keys = []
    for i in range(num_hidden_layers):
        rename_keys.extend(
            [
                (
                    f"text_model.encoder.layers.{i}.self_attn.k_proj.weight",
                    f"model.text_encoder.decoder.layers.{i}.self_attention.linear_k.weight",
                ),
                (
                    f"text_model.encoder.layers.{i}.self_attn.k_proj.bias",
                    f"model.text_encoder.decoder.layers.{i}.self_attention.linear_k.bias",
                ),
                (
                    f"text_model.encoder.layers.{i}.self_attn.q_proj.weight",
                    f"model.text_encoder.decoder.layers.{i}.self_attention.linear_q.weight",
                ),
                (
                    f"text_model.encoder.layers.{i}.self_attn.q_proj.bias",
                    f"model.text_encoder.decoder.layers.{i}.self_attention.linear_q.bias",
                ),
                (
                    f"text_model.encoder.layers.{i}.self_attn.v_proj.weight",
                    f"model.text_encoder.decoder.layers.{i}.self_attention.linear_v.weight",
                ),
                (
                    f"text_model.encoder.layers.{i}.self_attn.v_proj.bias",
                    f"model.text_encoder.decoder.layers.{i}.self_attention.linear_v.bias",
                ),
                (
                    f"text_model.encoder.layers.{i}.self_attn.out_proj.weight",
                    f"model.text_encoder.decoder.layers.{i}.self_attention.linear_proj.weight",
                ),
                (
                    f"text_model.encoder.layers.{i}.self_attn.out_proj.bias",
                    f"model.text_encoder.decoder.layers.{i}.self_attention.linear_proj.bias",
                ),
                (
                    f"text_model.encoder.layers.{i}.layer_norm1.weight",
                    f"model.text_encoder.decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight",
                ),
                (
                    f"text_model.encoder.layers.{i}.layer_norm1.bias",
                    f"model.text_encoder.decoder.layers.{i}.self_attention.linear_qkv.layer_norm_bias",
                ),
                (
                    f"text_model.encoder.layers.{i}.mlp.fc1.weight",
                    f"model.text_encoder.decoder.layers.{i}.mlp.linear_fc1.weight",
                ),
                (
                    f"text_model.encoder.layers.{i}.mlp.fc1.bias",
                    f"model.text_encoder.decoder.layers.{i}.mlp.linear_fc1.bias",
                ),
                (
                    f"text_model.encoder.layers.{i}.mlp.fc2.weight",
                    f"model.text_encoder.decoder.layers.{i}.mlp.linear_fc2.weight",
                ),
                (
                    f"text_model.encoder.layers.{i}.mlp.fc2.bias",
                    f"model.text_encoder.decoder.layers.{i}.mlp.linear_fc2.bias",
                ),
                (
                    f"text_model.encoder.layers.{i}.layer_norm2.weight",
                    f"model.text_encoder.decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight",
                ),
                (
                    f"text_model.encoder.layers.{i}.layer_norm2.bias",
                    f"model.text_encoder.decoder.layers.{i}.mlp.linear_fc1.layer_norm_bias",
                ),
                (
                    f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight",
                    f"model.vision_encoder.decoder.layers.{i}.self_attention.linear_k.weight",
                ),
                (
                    f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias",
                    f"model.vision_encoder.decoder.layers.{i}.self_attention.linear_k.bias",
                ),
                (
                    f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight",
                    f"model.vision_encoder.decoder.layers.{i}.self_attention.linear_v.weight",
                ),
                (
                    f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias",
                    f"model.vision_encoder.decoder.layers.{i}.self_attention.linear_v.bias",
                ),
                (
                    f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight",
                    f"model.vision_encoder.decoder.layers.{i}.self_attention.linear_q.weight",
                ),
                (
                    f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias",
                    f"model.vision_encoder.decoder.layers.{i}.self_attention.linear_q.bias",
                ),
                (
                    f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight",
                    f"model.vision_encoder.decoder.layers.{i}.self_attention.linear_proj.weight",
                ),
                (
                    f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias",
                    f"model.vision_encoder.decoder.layers.{i}.self_attention.linear_proj.bias",
                ),
                (
                    f"vision_model.encoder.layers.{i}.layer_norm1.weight",
                    f"model.vision_encoder.decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight",
                ),
                (
                    f"vision_model.encoder.layers.{i}.layer_norm1.bias",
                    f"model.vision_encoder.decoder.layers.{i}.self_attention.linear_qkv.layer_norm_bias",
                ),
                (
                    f"vision_model.encoder.layers.{i}.mlp.fc1.weight",
                    f"model.vision_encoder.decoder.layers.{i}.mlp.linear_fc1.weight",
                ),
                (
                    f"vision_model.encoder.layers.{i}.mlp.fc1.bias",
                    f"model.vision_encoder.decoder.layers.{i}.mlp.linear_fc1.bias",
                ),
                (
                    f"vision_model.encoder.layers.{i}.mlp.fc2.weight",
                    f"model.vision_encoder.decoder.layers.{i}.mlp.linear_fc2.weight",
                ),
                (
                    f"vision_model.encoder.layers.{i}.mlp.fc2.bias",
                    f"model.vision_encoder.decoder.layers.{i}.mlp.linear_fc2.bias",
                ),
                (
                    f"vision_model.encoder.layers.{i}.layer_norm2.weight",
                    f"model.vision_encoder.decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight",
                ),
                (
                    f"vision_model.encoder.layers.{i}.layer_norm2.bias",
                    f"model.vision_encoder.decoder.layers.{i}.mlp.linear_fc1.layer_norm_bias",
                ),
            ]
        )

    rename_keys.extend(
        [
            ("logit_scale", "model.logit_scale"),
            ("logit_bias", "model.logit_bias"),
            ("vision_model.embeddings.patch_embedding.weight", "model.vision_encoder.conv1.weight"),
            ("vision_model.embeddings.patch_embedding.bias", "model.vision_encoder.conv1.bias"),
            ("vision_model.embeddings.position_embedding.weight", "model.vision_encoder.position_embeddings.weight"),
            ("vision_model.post_layernorm.weight", "model.vision_encoder.final_layernorm.weight"),
            ("vision_model.post_layernorm.bias", "model.vision_encoder.final_layernorm.bias"),
            ("vision_model.head.probe", "model.vision_encoder.head.probe"),
            (
                "vision_model.head.attention.in_proj_weight",
                "model.vision_encoder.head.cross_attention.linear_qkv.weight",
            ),
            ("vision_model.head.attention.in_proj_bias", "model.vision_encoder.head.cross_attention.linear_qkv.bias"),
            (
                "vision_model.head.attention.out_proj.weight",
                "model.vision_encoder.head.cross_attention.linear_proj.weight",
            ),
            (
                "vision_model.head.attention.out_proj.bias",
                "model.vision_encoder.head.cross_attention.linear_proj.bias",
            ),
            ("vision_model.head.layernorm.weight", "model.vision_encoder.head.mlp.linear_fc1.layer_norm_weight"),
            ("vision_model.head.layernorm.bias", "model.vision_encoder.head.mlp.linear_fc1.layer_norm_bias"),
            ("vision_model.head.mlp.fc1.weight", "model.vision_encoder.head.mlp.linear_fc1.weight"),
            ("vision_model.head.mlp.fc1.bias", "model.vision_encoder.head.mlp.linear_fc1.bias"),
            ("vision_model.head.mlp.fc2.weight", "model.vision_encoder.head.mlp.linear_fc2.weight"),
            ("vision_model.head.mlp.fc2.bias", "model.vision_encoder.head.mlp.linear_fc2.bias"),
            ("text_model.embeddings.token_embedding.weight", "model.text_encoder.embedding.word_embeddings.weight"),
            (
                "text_model.embeddings.position_embedding.weight",
                "model.text_encoder.embedding.position_embeddings.weight",
            ),
            ("text_model.final_layer_norm.weight", "model.text_encoder.final_layernorm.weight"),
            ("text_model.final_layer_norm.bias", "model.text_encoder.final_layernorm.bias"),
            ("text_model.head.weight", "model.text_encoder.head.weight"),
            ("text_model.head.bias", "model.text_encoder.head.bias"),
        ]
    )

    return rename_keys


def rename_model_keys(model_state_dict, rename_keys):
    """
    Rename keys in the model's state dictionary based on the provided mappings.

    Parameters:
    model_state_dict (dict): The state dictionary of the model.
    rename_keys (list): A list of tuples with the mapping (old_key, new_key).

    Returns:
    dict: A new state dictionary with updated key names.
    """

    # Create a new state dictionary with updated key names
    new_state_dict = {}

    # Track keys from the original state dict to ensure all are processed
    remaining_keys = set(model_state_dict.keys())

    # Iterate over the rename mappings
    for old_key, new_key in rename_keys:
        if old_key in model_state_dict:
            # Rename the key and remove it from the tracking set
            new_state_dict[new_key] = model_state_dict[old_key]
            remaining_keys.remove(old_key)

    # Check if any keys were not converted from old to new
    for old_key in remaining_keys:
        print(f"Warning: Key '{old_key}' was not converted.")

    return new_state_dict


def adjust_tensor_shapes(model, nemo_state_dict):
    """
    Adapt tensor shapes in the state dictionary to ensure compatibility with a different model structure.

    Parameters:
    nemo_state_dict (dict): The state dictionary of the model.

    Returns:
    dict: The updated state dictionary with modified tensor shapes for compatibility.
    """
    model_config = model.cfg

    # Note: For 'key' and 'value' weight and biases, NeMo uses a consolidated tensor 'query_key_value'.
    for key_ in list(nemo_state_dict.keys()):
        if "vision" in key_:
            config = model_config["vision"]
        else:
            config = model_config["text"]
        num_query_groups = head_num = config["num_attention_heads"]
        hidden_size = config["hidden_size"]
        head_size = hidden_size // head_num
        heads_per_group = head_num // num_query_groups
        if "bias" in key_:
            hidden_size = 1

        if 'head.cross_attention.linear_qkv.' in key_:
            key_q = key_.replace('linear_qkv', 'linear_q')
            key_kv = key_.replace('linear_qkv', 'linear_kv')
            q_weight, k_weight, v_weight = nemo_state_dict[key_].chunk(3)
            k_weight = k_weight.reshape(num_query_groups, head_size, hidden_size)
            v_weight = v_weight.reshape(num_query_groups, head_size, hidden_size)
            kv_weight = torch.empty((0, head_size, hidden_size), device=q_weight.device)
            for i in range(num_query_groups):
                kv_weight = torch.cat((kv_weight, k_weight[i : i + 1, :, :]))
                kv_weight = torch.cat((kv_weight, v_weight[i : i + 1, :, :]))
            kv_weight = kv_weight.reshape([head_size * 2 * num_query_groups, hidden_size])
            if "bias" in key_:
                kv_weight = kv_weight.squeeze(-1)
            nemo_state_dict[key_q] = q_weight
            nemo_state_dict[key_kv] = kv_weight
            del nemo_state_dict[key_]

        if 'self_attention.linear_q.' in key_:
            key_q = key_
            key_k = key_.replace('linear_q', 'linear_k')
            key_v = key_.replace('linear_q', 'linear_v')
            key_qkv = key_.replace('linear_q', 'linear_qkv')

            # [(head_num + 2 * num_query_groups) * head_size, hidden_size]
            # -> [head_num, head_size, hidden_size], 2 * [num_query_groups, head_size, hidden_size]
            q_weight, k_weight, v_weight = nemo_state_dict[key_q], nemo_state_dict[key_k], nemo_state_dict[key_v]
            q_weight = q_weight.reshape(head_num, head_size, hidden_size)
            k_weight = k_weight.reshape(num_query_groups, head_size, hidden_size)
            v_weight = v_weight.reshape(num_query_groups, head_size, hidden_size)

            qkv_weight = torch.empty((0, head_size, hidden_size), device=q_weight.device)
            for i in range(num_query_groups):
                qkv_weight = torch.cat((qkv_weight, q_weight[i * heads_per_group : (i + 1) * heads_per_group, :, :]))
                qkv_weight = torch.cat((qkv_weight, k_weight[i : i + 1, :, :]))
                qkv_weight = torch.cat((qkv_weight, v_weight[i : i + 1, :, :]))
            qkv_weight = qkv_weight.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])
            if "bias" in key_:
                qkv_weight = qkv_weight.squeeze(-1)
            nemo_state_dict[key_qkv] = qkv_weight
            del nemo_state_dict[key_q], nemo_state_dict[key_k], nemo_state_dict[key_v]

    return nemo_state_dict


def adjust_nemo_config(model_config, ref_config):
    model_config["encoder_seq_length"] = ref_config["max_position_embeddings"]
    model_config["num_layers"] = ref_config["num_hidden_layers"]
    model_config["ffn_hidden_size"] = ref_config["intermediate_size"]
    model_config["hidden_size"] = ref_config["hidden_size"]
    model_config["num_attention_heads"] = ref_config["num_attention_heads"]
    model_config["num_query_groups"] = ref_config["num_key_value_heads"]
    model_config["kv_channels"] = ref_config["head_dim"]
    model_config["layernorm_epsilon"] = ref_config["rms_norm_eps"]
    return model_config


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--input_name_or_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            '../../examples/multimodal/vision_language_foundation/clip/conf/megatron_siglip_so400m_14_384.yaml',
        ),
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--output_path", type=str, default=None, help="Path to output .nemo file.")
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["bf16", "32"], help="Precision for checkpoint weight saved"
    )

    args = parser.parse_args()
    return args


def convert(args):
    logging.info(f"Loading checkpoint from HF: `{args.input_name_or_path}`")
    hf_model = AutoModel.from_pretrained(args.input_name_or_path)
    hf_processor = AutoProcessor.from_pretrained(args.input_name_or_path)
    logging.info("HF Model loading done.")

    nemo_config = OmegaConf.load(args.hparams_file)

    nemo_config.trainer["precision"] = args.precision
    trainer = MegatronTrainerBuilder(nemo_config).create_trainer()
    model = MegatronCLIPModel(nemo_config.model, trainer)

    assert nemo_config.model.text.num_layers == nemo_config.model.vision.num_layers
    rename_keys = create_rename_keys(nemo_config.model.text.num_layers)
    old_state_dict = hf_model.state_dict()
    new_state_dict = rename_model_keys(model_state_dict=old_state_dict, rename_keys=rename_keys)

    nemo_state_dict = adjust_tensor_shapes(model, new_state_dict)
    model.load_state_dict(nemo_state_dict, strict=False)

    logging.info(f'=' * 100)
    # Verifications
    import requests
    from PIL import Image

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    texts = ["a photo of 2 cats", "a photo of 2 dogs"]
    inputs = hf_processor(text=texts, images=image, padding="max_length", return_tensors="pt")

    tokens = inputs["input_ids"].cuda()
    text_model = model.model.text_encoder.cuda()
    hf_text_model = hf_model.text_model.cuda()
    text_model_output = text_model(tokens)
    hf_text_model_output = hf_text_model(tokens).pooler_output
    assert torch.allclose(text_model_output, hf_text_model_output, atol=0.01)
    logging.info(f'! Text model results matched.')

    pixels = inputs["pixel_values"].cuda()
    vision_model = model.model.vision_encoder.cuda()
    hf_vision_model = hf_model.vision_model.cuda()
    vision_model_output = vision_model(pixels)
    hf_vision_model_output = hf_vision_model(pixels).pooler_output
    assert torch.allclose(vision_model_output, hf_vision_model_output, atol=0.01)
    logging.info(f'! Vision model results matched.')

    logging.info(f'=' * 100)

    dtype = torch_dtype_from_precision(args.precision)
    model = model.to(dtype=dtype)
    model.save_to(args.output_path)
    logging.info(f'NeMo model saved to: {args.output_path}')


if __name__ == '__main__':
    args = get_args()
    convert(args)
