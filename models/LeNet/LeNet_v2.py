import torch
from datetime import datetime
from torch import nn
import torch.utils
from d2l import torch as d2l # pip install d2l
from icecream import ic
import time
from torchvision import datasets
import torch  
from torch.utils.data import Dataset, DataLoader  
from PIL import Image  
import os  
from torchvision import datasets, transforms  
from pathlib import Path  
from sklearn.model_selection import train_test_split # pip install scikit-learn 
from PIL import Image  
from torchvision import transforms  
import json

class Reshape(torch.nn.Module):
    def forward(self,x):
        return x.view(-1,1,28,28)

net = torch.nn.Sequential(
    Reshape(),
    nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Sigmoid(),  # torch.Size([6, 1, 5, 5])   torch.Size([6])
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),   # torch.Size([16, 6, 5, 5])     torch.Size([16])
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Flatten(),
    nn.Linear(16*5*5,300),nn.Sigmoid(),   # torch.Size([120, 400])  torch.Size([120])
    nn.Linear(300,120),nn.Sigmoid(),     # torch.Size([84, 120])   torch.Size([84])
    nn.Linear(120,62)            # torch.Size([10, 84])   torch.Size([10])
)


def train_ch6_(net,train_iter,test_iter,num_epochs,lr,device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m)==nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('train on', device)
    # net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],legend=['train loss','train acc','test acc'])
    timer,num_batches = d2l.Timer(),len(train_iter)
    # print(num_batches)
    for epoch in range(num_epochs):
        print('epoch:{}'.format(epoch),end='\t')
        metric = d2l.Accumulator(3)
        net.train()
        for i,(X,y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X,y = X.to(device),y.to(device)
            y_hat = net(X)
            l = loss(y_hat,y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l*X.shape[0],d2l.accuracy(y_hat,y),X.shape[0])
            timer.stop()
            train_l = metric[0]/metric[2]
            train_acc = metric[1]/metric[2]
            # if (i+1)%(num_batches//5)==0 or i==num_batches-1:
            #     animator.add(epoch+(i+1)/num_batches,(train_l,train_acc,None))
        test_acc = evaluate_accuracy_gpu(net,test_iter)
        # animator.add(epoch+1,(None,None,test_acc))
        print(f'loss {train_l:.3f},train acc {train_acc:.3f},'f'test acc {test_acc:.3f}')
        # 查看梯度
        for name, param in net.named_parameters():
            print(name,param.detach().shape,param.grad.shape)


def train_ch6(net,train_iter,test_iter,num_epochs,lr,device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m)==nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('train on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    # animator = d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],legend=['train loss','train acc','test acc'])
    # timer,num_batches = d2l.Timer(),len(train_iter)
    num_batches = len(train_iter)
    best_acc = 0
    # print(num_batches)
    for epoch in range(num_epochs):
        print('epoch:{}'.format(epoch),end='\t')
        metric = d2l.Accumulator(3)
        net.train()
        for i,(X,y) in enumerate(train_iter):
            # timer.start()
            optimizer.zero_grad()
            X,y = X.to(device),y.to(device)
            y_hat = net(X)
            # print(y_hat.shape)
            # print(y_hat)
            # print(y.shape)
            # print(y)
            # quit()
            l = loss(y_hat,y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l*X.shape[0],d2l.accuracy(y_hat,y),X.shape[0])
            # timer.stop()
            train_l = metric[0]/metric[2]
            train_acc = metric[1]/metric[2]
            # if (i+1)%(num_batches//5)==0 or i==num_batches-1:
            #     animator.add(epoch+(i+1)/num_batches,(train_l,train_acc,None))
        test_acc = evaluate_accuracy_gpu(net,test_iter)
        if test_acc>best_acc:
            weights_path = Path('./weights/')
            weights_path.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(),weights_path/Path('LeNetv2_best_weights.pth'))
        torch.save(net.state_dict(), weights_path/Path('LeNetv2_last_weights.pth'))
        # animator.add(epoch+1,(None,None,test_acc))
        print(f'loss {train_l:.3f},train acc {train_acc:.3f},'f'test acc {test_acc:.3f}')

def test(test_loader,weights,device):
    net.load_state_dict(torch.load(weights))
    net.to(device)
    net.eval()
    with open(data_root/Path('classes.txt'),'r',encoding='utf-8') as rf:
        classes = rf.read().split(',')
    with open(data_root/Path('classes.json'),'r',encoding='utf-8') as rf:
        classes_json=json.loads(rf.read())
    result = list()
    for data, name in test_loader:
        data = data.to(device)
        output=net(data)
        predicted_class = torch.argmax(output,dim=1) 
        for i,j in zip(name,predicted_class.tolist()):
            result.append('{},{}'.format(i,classes_json[classes[j]]))
    with open('result/LeNetbase_{}.csv'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")),'w',encoding='utf-8') as wf:
        wf.write('ImageID,label\n')
        wf.write('\n'.join(result))


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



def run():
    # 将训练集分为训练集和验证集
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])  
    train = datasets.ImageFolder(root='traffic_sign/dataset/train_set', transform=transform)
    # FashionMNIST_test =  datasets.ImageFolder(root='traffic_sign/dataset/test_set', transform=transform)
    # train_iter = DataLoader(dataset=FashionMNIST_train,batch_size=256)
    # test_iter = DataLoader(dataset=FashionMNIST_test,batch_size=256)
    # lr, num_epochs = 0.9, 5
    # train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

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


def split_dataset(datasets_root,data_root,exist_ok=True):
    data_root.mkdir(parents=True, exist_ok=True)
    '''
    将训练集按照1:9划分为训练集和验证集，并将训练集、验证集、测试集以地址形式存储在项目中
    当 exist_ok==True 时，若已存在文件，则不重新生成；若不存在则生成
    当 exist_ok==False 时，生成
    将类别信息存储在classes.txt中
    '''
    if exist_ok and (data_root/Path('train.csv')).exists() and (data_root/Path('val.csv')).exists() and (data_root/Path('test.csv')).exists() and (data_root/Path('classes.csv')).exists():
        return
    train_path = datasets_root/Path('train_set')
    test_path = datasets_root/Path('test_set')
    train_data_X, train_data_y,classes = list(), list(), list()
    for item in train_path.rglob('*'):
        if item.is_file():
            train_data_X.append(item.relative_to(datasets_root))
            train_data_y.append(item.parent.name)
        elif item.is_dir():
            classes.append(item.name)
    X_train, X_val, y_train, y_val = train_test_split(train_data_X, train_data_y, test_size=0.1, random_state=42)
    X_test = [item.relative_to(datasets_root) for item in test_path.rglob('*') if item.is_file()]
    with open(data_root/Path('classes.txt'),'w',encoding='utf-8') as wf:
        wf.write(','.join(classes))
    with open(data_root/Path('train.csv'),'w',encoding='utf-8') as wf:
        wf.write('\n'.join(['{},{}'.format(i,j) for i,j in zip(X_train,y_train)]))
    with open(data_root/Path('val.csv'),'w',encoding='utf-8') as wf:
        wf.write('\n'.join(['{},{}'.format(i,j) for i,j in zip(X_val,y_val)]))
    with open(data_root/Path('test.csv'),'w',encoding='utf-8') as wf:
        wf.write('\n'.join([str(i) for i in X_test]))



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
        img = img.convert('L')  
        resize = transforms.Resize((28, 28))  
        img = resize(img)  
        to_tensor = transforms.ToTensor()
        img = to_tensor(img) 
        if self.data_type == 'test':
            y = sample[1]
        else:
            y = self.classes.index(sample[1])
        return img,y

if __name__ == '__main__':
    datasets_root = Path('../data/traffic_sign/')
    data_root = Path('./data/traffic_sign/split_1/')
    #dataset(datasets_root)
    #split_dataset(datasets_root,data_root)
    train_loader = DataLoader(dataset=TrafficSign(datasets_root,data_root,'train'),batch_size=256,shuffle=True)
    val_loader = DataLoader(dataset=TrafficSign(datasets_root,data_root,'val'),batch_size=256,shuffle=True)
    #test_loader = DataLoader(dataset=TrafficSign(datasets_root,data_root,'test'),batch_size=256)
    lr, num_epochs = 1, 200
    train_ch6(net, train_loader, val_loader, num_epochs, lr, d2l.try_gpu())
    #test(test_loader,'weights/LeNetbase_best_weights.pth',d2l.try_gpu())
