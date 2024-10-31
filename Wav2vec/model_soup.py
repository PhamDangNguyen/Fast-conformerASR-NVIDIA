import torch
import torch.nn as nn
import copy

def soup_two_models(model, second_model):
    souped_model = copy.deepcopy(model)
    for param in souped_model.named_parameters():
        name = param[0]
        param[1].data = (model.state_dict()[name] + second_model.state_dict()[name]) / 2

    return souped_model

paths = ['/home/ndanh/asr-wav2vec/output/wav2vec2-large-nguyenvulebinh-25-12/checkpoint-18820000/pytorch_model.bin','/home/ndanh/asr-wav2vec/output/checkpoint-13340000/pytorch_model.bin']

weight_0 = torch.load(paths[0],map_location="cpu")
weight_1 = torch.load(paths[1],map_location="cpu")

weight_soup = {}
for key,value in weight_0.items():
    weight_soup[key] = value/2 + weight_1[key]/2

torch.save(weight_soup,"output/checkpoint-17120000_13340000/pytorch_model.bin")