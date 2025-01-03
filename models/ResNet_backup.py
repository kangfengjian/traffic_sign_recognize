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

def train_ch6(train_iter,test_iter,num_epochs,lr,device,model_name,index):
    def init_weights(m):
        if type(m) == nn.Linear or type(m)==nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net = nn.Sequential(b1, b2, b3, b4, b5,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(), nn.Linear(512, 62))
    net.apply(init_weights)
    print('train on', device)
    # print('模型yi转移到GPU',torch.cuda.memory_allocated()//1024//1024)
    net.to(device)
    # print('转移zyi完成',torch.cuda.memory_allocated()//1024//1024)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch',xlim=[1,num_epochs],legend=['train loss','train acc','test acc'],model_name=model_name,index=index)
    # timer,num_batches = d2l.Timer(),len(train_iter)
    num_batches = len(train_iter)
    best_acc = 0
    # print(num_batches)
    with open('log/log_{}_{}.log'.format(model_name,index),'w',encoding='utf-8') as wf:
        wf.write('')
    for epoch in range(num_epochs):
        start_time = time.perf_counter()
        out_str = '{}-cross_{}_epoch:{}/{}\t'.format(model_name,index,epoch,num_epochs)
        # print('epoch:{}'.format(epoch),end='\t')
        metric = d2l.Accumulator(3)
        net.train()
        for i,(X,y) in enumerate(train_iter):
            # timer.start()
            optimizer.zero_grad()
            # print('开始转移本batch的数据',torch.cuda.memory_allocated()//1024//1024)
            X,y = X.to(device),y.to(device)
            # print('本batch的数据转移完成',torch.cuda.memory_allocated()//1024//1024)
            y_hat = net(X)
            # print('本batch获得预测结果',torch.cuda.memory_allocated()//1024//1024)
            # print(y_hat.shape)
            # print(y_hat)
            # print(y.shape)
            # print(y)
            # quit()
            l = loss(y_hat,y)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                metric.add(l*X.shape[0],d2l.accuracy(y_hat,y),X.shape[0])
            # timer.stop()
            train_l = metric[0]/metric[2]
            train_acc = metric[1]/metric[2]
            if (i+1)%(num_batches//5)==0 or i==num_batches-1:
                animator.add(epoch+(i+1)/num_batches,(train_l,train_acc,None))
        test_acc = evaluate_accuracy_gpu(net,test_iter)
        if test_acc>best_acc:
            best_acc = test_acc
            weights_path = Path('./weights/')
            weights_path.mkdir(parents=True, exist_ok=True)
            if best_acc>0.0:
                torch.save(net.state_dict(),weights_path/Path('{}_{}_best_weights.pth'.format(model_name,index)))
        # torch.save(net.state_dict(), weights_path/Path('{}_{}_last_weights.pth'.format(model_name,index)))
        animator.add(epoch+1,(None,None,test_acc))
        with open('log/log_{}_{}.log'.format(model_name,index),'a',encoding='utf-8') as wf:
            wf.write(out_str+'\n')
        # print('本epoch结束',torch.cuda.memory_allocated()//1024//1024)
        process_time = time.perf_counter()-start_time
        out_str+=f'loss {train_l:.3f},train acc {train_acc:.3f},'f'test acc {test_acc:.3f},'f'time {process_time:.3f}s'
        print(out_str)

    
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
    def __init__(self, datasets_root, data_root, data='train', lable=True):  
        self.data_type = data
        if data=='train':
            with open(data_root/Path('train.csv'),'r',encoding='utf-8') as rf:
                self.data = [i.split(',') for i in rf.read().split('\n')]
        elif data=='val':
            with open(data_root/Path('val.csv'),'r',encoding='utf-8') as rf:
                self.data = [i.split(',') for i in rf.read().split('\n')]
        elif data=='test':
            with open(data_root/Path('test.csv'),'r',encoding='utf-8') as rf:
                self.data = [[i,Path(i).name] for i in rf.read().split('\n')]
        elif data=='test_B':
            with open(data_root/Path('test_B.csv'),'r',encoding='utf-8') as rf:
                self.data = [[i,Path(i).name] for i in rf.read().split('\n')]
        else:
            assert '未知data类型：{}'.format(data) 
        with open(data_root/Path('classes.txt'),'r',encoding='utf-8') as rf:
            self.classes = rf.read().split(',')
        self.datasets_root = datasets_root
  
    def __len__(self):  
        return len(self.data)
  
    def __getitem__(self, idx):  
        sample = self.data[idx]
        img = Image.open(self.datasets_root/Path(sample[0]))
        # 图片处理
        # img = img.convert('RGB')  
        img = img.convert('L')
        resize = transforms.Resize((96, 96))  
        img = resize(img)  
        to_tensor = transforms.ToTensor()
        img = to_tensor(img) 
        if self.data_type == 'test':
            y = sample[1]
        else:
            y = self.classes.index(sample[1])
        return img,y
    
def test(test_loader,weights,device,model_name,index):
    net = nn.Sequential(b1, b2, b3, b4, b5,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(), nn.Linear(512, 62))
    net.load_state_dict(torch.load(weights,weights_only=True))
    net.to(device)
    net.eval()
    # print('test模型加载完毕',torch.cuda.memory_allocated()//1024//1024,torch.cuda.memory_reserved()//1024//1024)
    with open(data_root/Path('classes.txt'),'r',encoding='utf-8') as rf:
        classes = rf.read().split(',')
    with open(data_root/Path('classes.json'),'r',encoding='utf-8') as rf:
        classes_json=json.loads(rf.read())
    result = list()
    for data, name in test_loader:
        # print('test数据准备加载',torch.cuda.memory_allocated()//1024//1024,torch.cuda.memory_reserved()//1024//1024)
        data = data.to(device)
        # print('test数据加载完成',torch.cuda.memory_allocated()//1024//1024,torch.cuda.memory_reserved()//1024//1024)
        output=net(data)
        # print('test模型预测完成',torch.cuda.memory_allocated()//1024//1024,torch.cuda.memory_reserved()//1024//1024)
        predicted_class = torch.argmax(output,dim=1) 
        for i,j in zip(name,predicted_class.tolist()):
            result.append('{},{}'.format(i,classes_json[classes[j]]))
    with open('result/{}_cross_{}_{}.csv'.format(model_name,index,datetime.now().strftime("%Y-%m-%d_%H:%M:%S")),'w',encoding='utf-8') as wf:
        wf.write('ImageID,label\n')
        wf.write('\n'.join(result))
    print('测试完成',torch.cuda.memory_allocated()//1024//1024)

if __name__ == '__main__':
    # print('strat...',torch.cuda.memory_allocated()//1024//1024)
    datasets_root = Path('../data/traffic_sign/')
    data_root = Path('./data/traffic_sign/split_1/')
    split_dataset(datasets_root,data_root)
    dataset(datasets_root)
    batch_size = 512
    model_name = 'ResNet-0000'
    parser = argparse.ArgumentParser(description='')  
    parser.add_argument('--index', type=str, help='交叉验证的划分号',default=0)
    args = parser.parse_args()
    cross_index = int(args.index)
    # print('加载训练集和验证集',torch.cuda.memory_allocated()//1024//1024)
    # train_loader = DataLoader(dataset=TrafficSign(datasets_root,data_root,'train'),batch_size=batch_size,shuffle=True)
    # val_loader = DataLoader(dataset=TrafficSign(datasets_root,data_root,'val'),batch_size=batch_size,shuffle=True)
    lr, num_epochs = 0.05, 20
    # train_ch6( train_loader, val_loader, num_epochs, lr, d2l.try_gpu(), model_name, cross_index)
    print('开始测试_{}'.format(cross_index),torch.cuda.memory_allocated()//1024//1024)
    test_loader = DataLoader(dataset=TrafficSign(datasets_root,data_root,'test_B'),batch_size=batch_size)
    test(test_loader,'weights/{}_{}_best_weights.pth'.format(model_name,cross_index),d2l.try_gpu(), model_name,cross_index)
