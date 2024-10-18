
# Performance Benchmarks

## Large Language Models**

### Pretraining

- The results in the table below show pre-training performance for various tasks at FP8 precision.
  - Container: [NeMo24.07](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags)
  - System: DGX-H100

| Model         | #-GPUs | GBS  | MBS | Sequence Length| TP | PP | CP | VP | Tokens / sec / GPU | Model TFLOP / sec / GPU | ***Est. time to train in days (10T tokens, 1K GPUs)*** |
| -----         | ------ | ---  | --- | ---------------| -- | -- | -- | -- | ------------------ | ----------------------- | ------------------------------------------------------ |
| GPT3-5B       | 64     | 2048 | 4   | 2048           | 1  | 1  | 1  | 1  | 23406              | 765                     | ***5***                                                |
| GPT3-20B      | 64     | 256  | 2   | 2048           | 2  | 1  | 1  | 1  | 5851               | 750                     | ***19***                                               |
| GPT3-175B     | 128    | 256  | 1   | 2048           | 4  | 8  | 1  | 6  | 716                | 771                     | **158**                                                |
| GPT3-175B     | 512    | 2048 | 2   | 2048           | 4  | 8  | 1  | 6  | 825                | [888](https://mlcommons.org/benchmarks/training/)                     | **137**                                                |
| LLAMA2-7B     | 8      | 128  | 1   | 4096           | 1  | 1  | 1  | 1  | 16934              | 780                     | ***7***                                                | 
| LLAMA2-13B    | 16     | 128  | 1   | 4096           | 1  | 4  | 1  | 10 | 8715               | 760                     | ***13***                                               |
| LLAMA2-70B    | 64     | 128  | 1   | 4096           | 4  | 4  | 1  | 20 | 1728               | 768                     | ***65***                                               |
| Nemotron-8B   | 64     | 256  | 4   | 4096           | 2  | 1  | 1  | 1  | 12507              | 643                     | ***9***                                                |
| Nemotron-22B  | 64     | 256  | 2   | 4096           | 2  | 4  | 1  | 10 | 4312               | 562                     | ***26***                                               |
| Nemotron-340B | 128    | 32   | 1   | 4096           | 8  | 8  | 1  | 12 | 326                | 686                     | ***347***                                              |
| LLAMA3-8B     | 8      | 128  | 1   | 8192           | 1  | 1  | 2  | 1  | 12273              | 711                     | ***9***                                                |
| LLAMA3-70B    | 64     | 128  | 1   | 8192           | 4  | 4  | 2  | 5  | 1524               | 734                     | ***74***                                               |

### Finetuning

- The results in the table below show finetuning performance of LLAMA2 models with SFT (supervised fine-tuning), and LoRA (Low-rank adaptors) at FP8 precision.
  - Container: [NeMo24.07](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags)
  - System: DGX-H100
- For fine-tuning, we use `SQuAD-v1.1 <https://rajpurkar.github.io/SQuAD-explorer/>`__ dataset, and the inputs are packed to 4096 tokens.


| Model      | Task     | #-GPUs | GBS | MBS | Packed Sequence Length | TP | PP | Tokens / sec / GPU | Model TFLOP / sec / GPU |  ***Est. time to finetune in mins (10M tokens)***  |
| -----      | ----     | ---    | --- | --- | --------------- | -- | -- | ------------------ | -----------------------        | -------------------------------------------------- |
| LLAMA2-7B  | SFT      | 8      | 32  | 1   | 4096            | 1  | 1  | 16891              | 673                            | ***1.2***                                          |
| LLAMA2-13B | SFT      | 8      | 32  | 1   | 4096            | 1  | 4  | 10176              | 787                            | ***2.0***                                          |
| LLAMA2-70B | SFT      | 16     | 32  | 1   | 4096            | 4  | 4  | 1816               | 749                            | ***5.7***                                          |
| LLAMA2-7B  | LoRA     | 8      | 32  | 1   | 4096            | 1  | 1  | 24824              | 663                            | ***0.8***                                          |
| LLAMA2-13B | LoRA     | 8      | 32  | 1   | 4096            | 1  | 1  | 14629              | 757                            | ***1.4***                                          |
| LLAMA2-70B | LoRA     | 8      | 32  | 1   | 4096            | 2  | 4  | 2621               | 722                            | ***7.9***                                          |
