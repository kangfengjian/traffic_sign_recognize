import threading
import os,time
from icecream import ic
import subprocess

process_num = 0

def doubler(args):
    os.system('python train.py --index {}'.format(args))
    global process_num
    process_num=process_num-1
    
def get_free_gpu():
	# os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        # output = subprocess.check_output('nvidia-smi -q -d Memory')
        output = subprocess.check_output(['nvidia-smi', '-q', '-d', 'Memory']).decode('utf-8')
        output=output.strip().split()
        return int(output[40])

if __name__ == '__main__':
    args = range(10)
    max_process=5
    memory_occupy = 3000 # 进程需要占用的GPU空间，单位M
    for i in args:
        free_gpu = get_free_gpu()
        while free_gpu<memory_occupy or process_num>=max_process:
            time.sleep(10)
            # print(process_num)
            free_gpu = get_free_gpu()
        my_thread = threading.Thread(target=doubler, args=(i,))
        my_thread.start()
        process_num+=1
        time.sleep(10)

