import torch
from torch import nn
from d2l import torch as d2l
from pathlib import Path
from torchvision import datasets, transforms  
from sklearn.model_selection import train_test_split # pip install scikit-learn 
from torch.utils.data import Dataset, DataLoader  
from PIL import Image  
import json
from datetime import datetime
import argparse 
from torch.nn import functional as F
import time
import mimetypes
from models.ResNet import ResNet

class TrafficSign(Dataset):  
    def __init__(self, data_source, dataset_root):
        if data_source.is_file(): # 判断输入是不是文件
            if is_binary_file(data_source): # 如果是二进制文件，默认是输入一张图片
                pass
            else: # 如果不是二进制文件，默认是一个存储图片路径的文档
                with open(data_source,'r',encoding='utf-8') as rf:
                    self.data = [[i.split(',')[0],Path(i.split(',')[0]).name] for i in rf.read().split('\n') if i]
                # print(self.data)
        else: # 如果是一个文件夹，默认是存储图片的文件夹
            pass
        self.dataset_root = dataset_root
  
    def __len__(self):  
        return len(self.data)
  
    def __getitem__(self, idx):  
        sample = self.data[idx]
        img = Image.open(self.dataset_root/Path(sample[0]))
        # 图片处理
        img = img.convert('RGB')  
        # img = img.convert('L')
        resize = transforms.Resize((96, 96))  
        img = resize(img)  
        to_tensor = transforms.ToTensor()
        img = to_tensor(img) 
        y = sample[1]
        return img,y



def is_binary_file(file_path):
    mime_type, encoding = mimetypes.guess_type(file_path)
    return mime_type is None or mime_type.startswith('application/') or 'image' in mime_type


if __name__ == '__main__':
    device = d2l.try_gpu()
    print('infer on {}'.format(device))
    # 首先加载数据，数据应该是csv的，存储样本和标签的信息,也可以是一个文件夹
    data_source = Path('./data/traffic_sign/test_B.csv')
    datasets_root = Path('../data/traffic_sign/')
    data_root = Path('./data/traffic_sign/')
    batch_size = 512
    dataset = TrafficSign(data_source, datasets_root)
    test_loader = DataLoader(dataset=dataset,batch_size=batch_size)
    # 确定是交叉验证的哪一个
    parser = argparse.ArgumentParser(description='')  
    parser.add_argument('--index', type=str, help='交叉验证的划分号',default=0)
    cross_index = int(parser.parse_args().index)
    # 加载模型和训练好的参数\
    model_name = 'ResNet-B-0001'
    net  = ResNet
    weights = 'weights/{}_{}_best_weights.pth'.format(model_name,cross_index)
    net.load_state_dict(torch.load(weights,weights_only=True))
    net.to(device)
    net.eval()
    # 开始推理
    result = list()
    sample_num, right_num = 0,0
    for data, name in test_loader:
        data = data.to(device)
        output=net(data)
        predicted_class = torch.argmax(output,dim=1)
        for i,j in zip(name,predicted_class.tolist()):
            result.append([str(i),str(j)])
    with open('result/{}_cross_{}_{}.csv'.format(model_name,cross_index,datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),'w',encoding='utf-8') as wf:
        wf.write('ImageID,label\n')
        for i in result:
            wf.write(','.join(i)+'\n')
    print('{} infer over'.format(cross_index))
