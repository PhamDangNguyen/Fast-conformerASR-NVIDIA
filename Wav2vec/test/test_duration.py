import pandas as pd
import random
import json
import tqdm
import os
import re

def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)

if __name__ == '__main__':
    total_time = 0
    for file in recursive_walk('dataset'):
        if '_val.csv' in file:
            data = pd.read_csv(file)
            duration = data['duration'].sum()
            print(os.path.basename(file), ' => duration:', duration)
            total_time += duration

    print('prediction time: 700 (s)')
    print(f'duration: {total_time} (s)')
    print('average processing time:', 700.0 / total_time)
