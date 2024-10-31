"""
Kiểm tra lỗi path audio có space khiến train bị lỗi
"""

import os
import json
import random
import tqdm


def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)
    
    
if __name__ == '__main__':
    
    csv_name = './duyanh_new_augment_08_2023.csv'
    
    with open(f'duyanh_new_augment_08_2023_train.csv', 'w') as fp:
        print(f'file,text,duration', file=fp)
        for line in tqdm.tqdm(open(csv_name).read().strip().split('\n')):
            path, text, duration = line.split(',')
            if ' ' in path:
                print('Path:', path)
                continue
            
            if float(duration) >= 20:
                print('Duration:', duration)
                continue   
            
            if len(text) < 1:
                print('Text', text)
                continue
            
            path = path.replace('50.duyanh_augment_08_2023/', '/media/storage/hai/dataset/50.duyanh_augment_08_2023/')
            assert os.path.exists(path), print(path)
            
            print(f'{path},{text},{duration}', file=fp)