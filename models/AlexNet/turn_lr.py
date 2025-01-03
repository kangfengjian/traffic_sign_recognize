import threading
import os,time
from icecream import ic

def doubler(args):
    os.system('cd /fs/sf;python --index 4')
    
def get_free_gpu():
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
	memory_gpu = [int(x.split()[2]) for x in open('tmp','r').readlines()]
	if not memory_gpu:
		return -1
	gpuFree = max(memory_gpu)<16000
	if gpuFree:
		return -1
	else:
		return memory_gpu.index(max(memory_gpu))

if __name__ == '__main__':
    args = range(10)
    memory_occupy = 2000 # 进程需要占用的GPU空间，单位M
    for i in args:
        device = get_free_gpu()
        while device==-1 or lock<=0:
            time.sleep(20)
            device = get_free_gpu()
        lock = lock-1
        my_thread = threading.Thread(target=doubler, args=(section, dataset, kfold_idx, batch_size, hidden_dim, epochs, device))
        my_thread.start()
        time.sleep(15)

