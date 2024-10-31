import requests
import random
import numpy
import tqdm
import json
import sys
import os
import re
import soundfile as sf
import numpy as np
import random
import kenlm
import torch
import time
import pandas as pd
import argparse

def edit_distance(r, h):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d

def get_step_list(r, h, d):
    '''
    This function is to get the list of steps in the process of dynamic programming.
    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calulating the editting distance of h and r.
    '''
    x = len(r)
    y = len(h)
    list = []
    while True:
        if x == 0 and y == 0: 
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1] and r[x-1] == h[y-1]: 
            list.append("e")
            x = x - 1
            y = y - 1
        elif y >= 1 and d[x][y] == d[x][y-1]+1:
            list.append("i")
            x = x
            y = y - 1
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1]+1:
            list.append("s")
            x = x - 1
            y = y - 1
        else:
            list.append("d")
            x = x - 1
            y = y
    return list[::-1]

def wer(refs, hyps, use_tqdm=True):
    import tqdm
    nom = 0
    denom = 0
    if use_tqdm:
        bar = tqdm.tqdm(range(len(refs)))
    else:
        bar = range(len(refs))
    for i in bar:
        r = refs[i].split()
        h = hyps[i].split()
        d = edit_distance(r, h)
        nom += d[len(r)][len(h)]
        denom += len(r)
    result = float(nom) / denom * 100
    # result = str("%.2f" % result) + "%"
    return result
  
def test():
    r = ['chó thành tính bẩn nhỉ', 'chó hải học dốt thế nhỉ']
    h = ['chó thành tính sạch nhỉ', 'chó hải học dốt thế nhờ']
    result = wer(r, h)   
    print('WER:', result)


def get_wer(output_name, delimiter=','):
    # bar = tqdm.tqdm(pd.read_csv(output_name, header=None).values)
    lines = [line.split(delimiter) for line in open(output_name).read().strip().split('\n')]
    refs = []
    hyps = []
    for (label, predict) in lines:
        if len(label.split(' ')) < 2 or len(predict.split(' ')) < 2:
            continue
        refs.append(label)
        hyps.append(predict)

    result = wer(refs, hyps) 
    print('WER:', result)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str)
    args = parser.parse_args()

    get_wer(args.csv)
