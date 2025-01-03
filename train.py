import torch
from torch import nn
from d2l import torch as d2l
from pathlib import Path
from torchvision import transforms  
from torch.utils.data import Dataset, DataLoader  
from PIL import Image  
import argparse 
import time
from models.ResNet import ResNet
import imgaug.augmenters as iaa
import numpy as np

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




        
class TrafficSign(Dataset):  
    def __init__(self, data_file, dataset_root):  
        self.transforms = [
            lambda x:x,
            transforms.RandomHorizontalFlip(),  # 左右翻转
            transforms.RandomVerticalFlip(),  # 垂直翻转
            transforms.RandomRotation((45,45)),   # 旋转45度
            transforms.RandomRotation((90,90)),   # 旋转45度
            transforms.RandomRotation((135,135)),   # 旋转45度
            transforms.RandomRotation((180,180)),   # 旋转45度
            transforms.RandomRotation((225,225)),   # 旋转45度
            transforms.RandomRotation((270,270)),   # 旋转45度
            transforms.RandomRotation((315,315)),   # 旋转45度
            transforms.RandomAffine(0,translate=(0.3,0.3)),  # 平移
            transforms.RandomAffine(0,translate=(0.3,0.3)),  # 平移
            transforms.RandomAffine(0,translate=(0.3,0.3)),  # 平移
            transforms.RandomAffine(0,translate=(0.3,0.3)),  # 平移
            lambda x:transforms.Resize((x.size[0], x.size[1]//2))(x),   # 横向缩放
            lambda x:transforms.Resize((x.size[0]//2, x.size[1]))(x),   # 纵向缩放
            transforms.GaussianBlur(7, 3),  # 高斯模糊
            lambda x:Image.fromarray(iaa.MotionBlur(k=15, angle=0)(image = np.array(x)[:, :, ::-1] )),  # 运动模糊
            lambda x:transforms.Resize((30, 30))(transforms.ColorJitter(brightness=0.1, contrast=0.1)(x)), # 降低亮度并缩小
            ]
        with open(data_file,'r',encoding='utf-8') as rf:
            self.data = list()
            for pic in [i.split(',') for i in rf.read().split('\n') if i]:
                for j in range(len(self.transforms)):
                    if int(pic[1]) in [21,23,25,26,27,28,33,34,50,53,55] and j in [1]:
                        continue
                    self.data.append([pic[0]+'_{}'.format(j),pic[1]])
        self.dataset_root = dataset_root
  
    def __len__(self):  
        return len(self.data)
  
    def __getitem__(self, idx):  
        sample = self.data[idx]
        img = Image.open(self.dataset_root/Path(sample[0][:sample[0].rfind('_')])) # 打开图片，
        img = img.convert('RGB')   # 首先统一为RGB的形式，然后进行处理
        img = self.transforms[int(sample[0][sample[0].rfind('_')+1:])](img) # 数据增强处理  
        img = img.convert('L') # 改为单通道
        img = transforms.Resize((96, 96))(img) # 统一大小
        img = transforms.ToTensor()(img) 
        y = int(sample[1])
        return img,y
    
def process_bar(progress,total):
        # 计算当前进度条的百分比  
        percent = (progress / total) * 100  
        # 构造进度条字符串，使用空格来填充剩余部分  
        bar = f'[{">" * int(percent)}' + ' ' * (100 - int(percent)) + f']'  +' {}/{}'.format(progress,total)
        # 使用\r回到行首，然后打印进度条  
        print(f'\r{bar}', end='', flush=True)    

def init_weights(m):
    if type(m) == nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

if __name__ == '__main__':
    device = d2l.try_gpu()
    print('train on {}'.format(device))
    # 首先加载数据，数据应该是csv的，存储样本和标签的信息
    print('load data...')
    datasets_root = Path('../data/traffic_sign/')
    data_root = Path('./data/traffic_sign_B/')
    # 确定是交叉验证的哪一个
    parser = argparse.ArgumentParser(description='')     
    parser.add_argument('--index', type=str, help='交叉验证的划分号',default=0)
    cross_index = int(parser.parse_args().index)
    ## 加载数据
    batch_size = 512
    train_dataset = TrafficSign(data_root/'train_{}.csv'.format(cross_index),datasets_root)
    train_iter = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_dataset = TrafficSign(data_root/'test_{}.csv'.format(cross_index),datasets_root)
    test_iter = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)
    # num = 0
    # for i in test_iter:
    #     num+=i[0].shape[0]
    # print(num)
    # quit()
    # 构建模型
    print('generate model...')
    lr, num_epochs = 0.05, 5
    net  = ResNet
    net.apply(init_weights)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    # 开始训练
    print('start train...')
    model_name = 'ResNet-B-0002'
    animator = Animator(xlabel='epoch',xlim=[1,num_epochs],legend=['train loss','train acc','test acc'],model_name=model_name,index=cross_index)
    num_batches = len(train_iter)
    best_acc = 0
    with open('log/log_{}_{}.log'.format(model_name,cross_index),'w',encoding='utf-8') as wf:
        wf.write('')
    for epoch in range(num_epochs):
        start_time = time.perf_counter()
        out_str = '{}-cross_{}_epoch:{}/{}\t'.format(model_name,cross_index,epoch,num_epochs)
        metric = d2l.Accumulator(3)
        net.train()
        for i,(X,y) in enumerate(train_iter):
            optimizer.zero_grad()
            X,y = X.to(device),y.to(device)
            y_hat = net(X)
            l = loss(y_hat,y)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                metric.add(l*X.shape[0],d2l.accuracy(y_hat,y),X.shape[0])
            train_l = metric[0]/metric[2]
            train_acc = metric[1]/metric[2]
            if (i+1)%(num_batches//5)==0 or i==num_batches-1:
                animator.add(epoch+(i+1)/num_batches,(train_l,train_acc,None))
            process_bar(i,num_batches)
        test_acc = evaluate_accuracy_gpu(net,test_iter)
        if test_acc>best_acc:
            best_acc = test_acc
            weights_path = Path('./weights/')
            weights_path.mkdir(parents=True, exist_ok=True)
            if best_acc>0.9:
                torch.save(net.state_dict(),weights_path/Path('{}_{}_best_weights.pth'.format(model_name,cross_index)))
        # torch.save(net.state_dict(), weights_path/Path('{}_{}_last_weights.pth'.format(model_name,index)))
        animator.add(epoch+1,(None,None,test_acc))
        with open('log/log_{}_{}.log'.format(model_name,cross_index),'a',encoding='utf-8') as wf:
            wf.write(out_str+'\n')
        process_time = time.perf_counter()-start_time
        out_str+=f'loss {train_l:.3f},train acc {train_acc:.3f},'f'test acc {test_acc:.3f},'f'time {process_time:.3f}s'
        print(f'\r'+' '*120, end = '', flush=True)  
        print(f'\r'+out_str)  
    print(weights_path/Path('{}_{}_best_weights.pth'.format(model_name,cross_index)))
