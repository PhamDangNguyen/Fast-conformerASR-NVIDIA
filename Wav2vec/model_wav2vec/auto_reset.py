"""
When train, maybe oom, need check to auto reset
Latest update: after 6/2023: dont need auto reset
"""
from subprocess import Popen, PIPE
import os
import time



# cmd_start = 'python3 train_base.py > log.txt'
cmd_stop = 'pkill -SIGINT -f train_base.py'

print('[+] start')
# os.system(cmd_start)
Popen(['python3', 'train_base.py'], stdout=PIPE, stderr=PIPE)

while True:
    for i in range(3600*4, 0, -10):
        print(f"Remain: {i} (seconds)", end="\r", flush=True)
        time.sleep(10)

    print('[+] stop')
    os.system(cmd_stop)
    time.sleep(20)
    print('[+] start')
    Popen(['python3', 'train_base.py'], stdout=PIPE, stderr=PIPE)
    # os.system(cmd_start)

