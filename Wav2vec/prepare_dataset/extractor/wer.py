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
import time
import pandas as pd

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

def aligned_print(list, r, h, result):
    '''
    This funcition is to print the result of comparing reference and hypothesis sentences in an aligned way.
    
    Attributes:
        list   -> the list of steps.
        r      -> the list of words produced by splitting reference sentence.
        h      -> the list of words produced by splitting hypothesis sentence.
        result -> the rate calculated based on edit distance.
    '''
    print("REF:", end=" ")
    for i in range(len(list)):
        if list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            print(" "*(len(h[index])), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) < len(h[index2]):
                print(r[index1] + " " * (len(h[index2])-len(r[index1])), end=" ")
            else:
                print(r[index1], end=" "),
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print(r[index], end=" "),
    print("\nHYP:", end=" ")
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print(" " * (len(r[index])), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
                print(h[index2] + " " * (len(r[index1])-len(h[index2])), end=" ")
            else:
                print(h[index2], end=" ")
        else:
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            print(h[index], end=" ")
    print("\nEVA:", end=" ")
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print("D" + " " * (len(r[index])-1), end=" ")
        elif list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            print("I" + " " * (len(h[index])-1), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
                print("S" + " " * (len(r[index1])-1), end=" ")
            else:
                print("S" + " " * (len(h[index2])-1), end=" ")
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print(" " * (len(r[index])), end=" ")
    print("\nWER: " + result)

def wer(refs, hyps, use_tqdm=True):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split()) 
    """
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

def read_config(filename):
    results = []
    with open(filename) as fid:
        for l in fid:
            try:
                results.append(json.loads(l))
            except:
                continue
    return results

def get_fpt_wer():
    ref_metas = json.loads(open('ref.json').read())

    metas = read_config('fpt.json')
    refs = []
    hyps = []
    print('construct dataset')
    count = 0
    for meta in tqdm.tqdm(metas):
        hyp = meta['predict']
        if re.findall(r'[0-9]+', hyp):
            count += 1
            continue
        hyp = re.sub(r'\_+', r' ', hyp)
        hyp = re.sub(r'\W+', r' ', hyp)
        hyp = re.sub(r' +', r' ', hyp)
        hyp = hyp.strip().lower()
        hyps.append(hyp)

        refs.append(ref_metas[meta['filename']]['text'])
        
    print('skip:', count)

    print('compute wer')
    result = wer(refs, hyps)   
    print('WER:', result)

def get_cmc_wer():
    metas = read_config('common_voice_basefull_noise_en_evaluate_18042022.json')
    refs = []
    hyps = []
    hyps_lm = []
    print('construct dataset')

    for meta in tqdm.tqdm(metas):
        refs.append(meta['text'])

        hyp = meta['predict']
        hyp = re.sub(r'\_+', r' ', hyp)
        hyp = re.sub(r'\W+', r' ', hyp)
        hyp = re.sub(r' +', r' ', hyp)
        hyp = hyp.strip().lower()
        hyps.append(hyp)

        # hyp = meta['predict_lm']
        # hyp = re.sub(r'\_+', r' ', hyp)
        # hyp = re.sub(r'\W+', r' ', hyp)
        # hyp = re.sub(r' +', r' ', hyp)
        # hyp = hyp.strip().lower()
        # hyps_lm.append(hyp)

    print('compute wer')
    result = wer(refs, hyps)   
    print('WER:', result)
    # result = wer(refs, hyps_lm)   
    # print('WER (+LM):', result)

def get_wer_common_voice(output_name):
    bar = tqdm.tqdm(pd.read_csv(output_name, header=None).values)
    refs = []
    hyps = []
    for item in bar:
        filename, label, predict = item
        
        label = label.strip()
        label = label.replace("  ","")
        label = label.lower()
        
        predict = predict.strip()
        predict = predict.replace("  ","")
        predict = predict.lower()
        if not filename.endswith('.wav'):
            continue

        if not isinstance(predict, str):
            continue
            
        if len(label.split(' ')) < 3 or len(predict.split(' ')) < 3:
            continue
        refs.append(label)
        hyps.append(predict)

    result = wer(refs, hyps)
    
    print(output_name, 'WER:', result)


def get_wer_common(output_name, delimiter=','):
    # bar = tqdm.tqdm(pd.read_csv(output_name, header=None).values)
    bar = tqdm.tqdm([line.split(delimiter) for line in open(output_name).read().strip().split('\n')])
    refs = []
    hyps = []
    for item in bar:
        filename, label, predict = item
        if not filename.endswith('.wav'):
            continue

        if not isinstance(predict, str):
            continue
            
        if len(label.split(' ')) < 5 or len(predict.split(' ')) < 5:
            continue
        refs.append(label)
        hyps.append(predict)

    result = wer(refs, hyps) 
    print('WER:', result)

if __name__ == '__main__':
    # output_name = 'raw_youtube_mien_trung.csv'
    # get_wer_common_voice(output_name)
    # path = 'Telesale3_09_2022_predict.csv'
    # get_wer_common(path, '|')
    test()
