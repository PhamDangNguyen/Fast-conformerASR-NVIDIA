import os
import csv

def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)


os.makedirs("dataset_cmc_remove_duplicate", exist_ok=True)
for mode in ['train','val']:

    csv_files = [f for f in recursive_walk('dataset_cmc') \
                            if f.endswith(f'_{mode}.csv')]
    
    for csv_file in csv_files:
        
        with open(f'{os.path.basename(csv_file)}', 'w') as fp:
            fp.write('file,text,duration\n')
            lines = open(csv_file).read().strip().split('\n')[1:]
            line_unique = set()
            num_duplicate = 0
            for line in lines:
                
                path, text, duration = line.split(',')
                if path not in line_unique:
                    line_unique.add(path)
                    print(f'{path},{text},{duration}', file=fp)
                else:
                    num_duplicate += 1
            
            print(f'{csv_file} has {num_duplicate} lines duplicates')


