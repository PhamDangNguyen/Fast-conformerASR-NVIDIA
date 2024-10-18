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

set -x

echo "unset all SLURM_, PMI_, PMIX_ Variables"
for i in $(env | grep ^SLURM_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMI_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMIX_ | cut -d"=" -f 1); do unset -v $i; done


python tests/export/nemo_export.py --model_name LLAMA2-7B-base --existing_test_models --min_tps 1 --max_tps 2
python tests/export/nemo_export.py --model_name LLAMA2-7B-base --existing_test_models --ptuning --min_tps 1 --max_tps 2
python tests/export/nemo_export.py --model_name LLAMA2-7B-base --existing_test_models --lora --min_tps 1 --max_tps 2
python tests/export/nemo_export.py --model_name LLAMA2-7B-code --existing_test_models --min_tps 1 --max_tps 2
python tests/export/nemo_export.py --model_name LLAMA2-7B-base-int4-awq --existing_test_models --min_tps 1 --max_tps 1
python tests/export/nemo_export.py --model_name LLAMA2-7B-base-int8-sq --existing_test_models --min_tps 1 --max_tps 1
python tests/export/nemo_export.py --model_name LLAMA2-7B-fp8-sft --existing_test_models --min_tps 1
python tests/export/nemo_export.py --model_name LLAMA2-13B-base --existing_test_models --min_tps 1 --max_tps 2
python tests/export/nemo_export.py --model_name LLAMA2-13B-base --existing_test_models --ptuning --min_tps 1 --max_tps 2
python tests/export/nemo_export.py --model_name LLAMA2-13B-base-fp8 --existing_test_models --min_tps 2 --max_tps 2
python tests/export/nemo_export.py --model_name LLAMA2-70B-base --existing_test_models --min_tps 2 --max_tps 8
python tests/export/nemo_export.py --model_name LLAMA2-70B-base-int4-awq --existing_test_models --min_tps 4 --max_tps 4
python tests/export/nemo_export.py --model_name LLAMA2-70B-base-int8-sq --existing_test_models --min_tps 2 --max_tps 2
python tests/export/nemo_export.py --model_name LLAMA3-8B-base-fp8 --existing_test_models --min_tps 1 --max_tps 1
python tests/export/nemo_export.py --model_name LLAMA3-70B-base-fp8 --existing_test_models --min_tps 8 --max_tps 8
python tests/export/nemo_export.py --model_name FALCON-7B-base --existing_test_models --min_tps 1 --max_tps 1
python tests/export/nemo_export.py --model_name FALCON-40B-base --existing_test_models --min_tps 2 --max_tps 8
python tests/export/nemo_export.py --model_name STARCODER1-15B-python --existing_test_models --min_tps 1 --max_tps 1
python tests/export/nemo_export.py --model_name STARCODER2-15B-4k-vfinal --existing_test_models --min_tps 1 --max_tps 1
python tests/export/nemo_export.py --model_name GEMMA-2B-base --existing_test_models --min_tps 1 --max_tps 1
python tests/export/nemo_export.py --model_name NEMOTRON3-22B-base-32k-v1 --existing_test_models --min_tps 2
python tests/export/nemo_export.py --model_name NEMOTRON3-22B-base-32k-v2 --existing_test_models --min_tps 2
python tests/export/nemo_export.py --model_name NEMOTRON3-22B-base-32k-v3 --existing_test_models --min_tps 2
python tests/export/nemo_export.py --model_name NEMOTRON4-340B-base-fp8 --existing_test_models --min_tps 8 --trt_llm_export_kwargs '{"reduce_fusion": false}'
