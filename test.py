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

class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(14, 10),model_name='',index=''):
        """Defined in :numref:`sec_utils`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        self.model_name=model_name
        self.index = index

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        self.fig.savefig('log/log_{}_{}.png'.format(self.model_name,self.index))

class Residual(nn.Module): #@save
    def __init__(self, input_channels, num_channels,use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),nn.BatchNorm2d(64), nn.ReLU(),nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

# net = nn.Sequential(b1, b2, b3, b4, b5,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(), nn.Linear(512, 10))

def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使⽤GPU计算模型在数据集上的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval() # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            # BERT微调所需的（之后将介绍）
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]



def split_dataset(datasets_root,data_root,exist_ok=True):
    test_path = datasets_root/Path('test_set_B')
    X_test = [item.relative_to(datasets_root) for item in test_path.rglob('*') if item.is_file()]
    with open(data_root/Path('test_B.csv'),'w',encoding='utf-8') as wf:
        wf.write('\n'.join([str(i) for i in X_test]))




def dataset(data_root):
    '''准备数据集
    '''
    data_root.mkdir(parents=True, exist_ok=True)
    # 检查是否存在训练集文件,没有则创建
    if not (data_root/Path('train_set')).exists():
        datasets.utils.download_and_extract_archive('https://ai-contest-static.xfyun.cn/2024/%E4%BA%A4%E9%80%9A%E6%A0%87%E8%AF%86%E8%AF%86%E5%88%AB%E6%8C%91%E6%88%98%E8%B5%9B/train_set.zip',data_root,data_root)
    if not (data_root/Path('test_set')).exists():
        datasets.utils.download_and_extract_archive('https://ai-contest-static.xfyun.cn/2024/%E4%BA%A4%E9%80%9A%E6%A0%87%E8%AF%86%E8%AF%86%E5%88%AB%E6%8C%91%E6%88%98%E8%B5%9B/test_set.zip',data_root,data_root)
    if not (data_root/Path('example.csv')).exists():
        datasets.utils.download_url('https://ai-contest-static.xfyun.cn/2024/%E4%BA%A4%E9%80%9A%E6%A0%87%E8%AF%86%E8%AF%86%E5%88%AB%E6%8C%91%E6%88%98%E8%B5%9B/example.csv',data_root)

class TrafficSign(Dataset):  
    def __init__(self, data_file,class_file, dataset_root,label=True):  
        if label:
            with open(data_file,'r',encoding='utf-8') as rf:
                self.data = [i.split(',') for i in rf.read().split('\n') if i]
        else:
            with open(data_file,'r',encoding='utf-8') as rf:
                self.data = [[i,Path(i).name] for i in rf.read().split('\n')] 
        with open(class_file,'r',encoding='utf-8') as rf:
            self.classes = rf.read().split(',')
        self.label = label
        self.dataset_root = dataset_root
  
    def __len__(self):  
        return len(self.data)
  
    def __getitem__(self, idx):  
        sample = self.data[idx]
        img = Image.open(self.dataset_root/Path(sample[0]))
        # 图片处理
        # img = img.convert('RGB')  
        img = img.convert('L')
        resize = transforms.Resize((96, 96))  
        img = resize(img)  
        to_tensor = transforms.ToTensor()
        img = to_tensor(img) 
        if not self.label:
            y = sample[1]
        else:
            y = self.classes.index(sample[1])
        return img,y

if __name__ == '__main__':
    device = d2l.try_gpu()
    print('test on {}'.format(device))
    # 首先加载数据，数据应该是csv的，存储样本和标签的信息
    datasets_root = Path('../data/traffic_sign/')
    data_root = Path('./data/traffic_sign/')
    batch_size = 512
    dataset = TrafficSign(data_file = data_root/'test.csv',class_file=data_root/'classes.txt',dataset_root=datasets_root)
    test_loader = DataLoader(dataset=dataset,batch_size=batch_size)
    #for i in test_loader:
    #    print(len(i),i[0].shape,i[1].shape)
    #    quit()
    # 加载模型和训练好的参数
    from models.ResNet import ResNet
    net  = ResNet
    weights = 'weights/ResNet-0000_0_best_weights.pth'
    net.load_state_dict(torch.load(weights,weights_only=True))
    net.to(device)
    net.eval()
    # 开始测试
    sample_num, right_num = 0,0
    for data, name in test_loader:
        data = data.to(device)
        output=net(data)
        predicted_class = torch.argmax(output,dim=1)
        for i,j in zip(name,predicted_class.tolist()):
            if i==j:
                right_num+=1
            sample_num+=1
    print('{}/{},{:.4f}'.format(right_num,sample_num,right_num/sample_num))

