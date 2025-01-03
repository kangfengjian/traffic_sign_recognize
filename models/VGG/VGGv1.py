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
        # display.display(self.fig)
        # display.clear_output(wait=True)



def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    conv_blks = []
    in_channels = 3
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 62))


# conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
# net = vgg(conv_arch)

# net = nn.Sequential(
#     # 这⾥，我们使⽤⼀个11*11的更⼤窗⼝来捕捉对象。
#     # 同时，步幅为4，以减少输出的⾼度和宽度。
#     # 另外，输出通道的数⽬远⼤于LeNet
#     nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
#     nn.MaxPool2d(kernel_size=3, stride=2),
#     # 减⼩卷积窗⼝，使⽤填充为2来使得输⼊与输出的⾼和宽⼀致，且增⼤输出通道数
#     nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
#     nn.MaxPool2d(kernel_size=3, stride=2),
#     # 使⽤三个连续的卷积层和较⼩的卷积窗⼝。
#     # 除了最后的卷积层，输出通道的数量进⼀步增加。
#     # 在前两个卷积层之后，汇聚层不⽤于减少输⼊的⾼度和宽度
#     nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
#     nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
#     nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
#     nn.MaxPool2d(kernel_size=3, stride=2),
#     nn.Flatten(),
#     # 这⾥，全连接层的输出数量是LeNet中的好⼏倍。使⽤dropout层来减轻过度拟合
#     nn.Linear(6400, 4096), nn.ReLU(),
#     nn.Dropout(p=0.5),
#     nn.Linear(4096, 4096), nn.ReLU(),
#     nn.Dropout(p=0.5),
#     # 最后是输出层。由于这⾥使⽤Fashion-MNIST，所以⽤类别数为10，⽽⾮论⽂中的1000
#     nn.Linear(4096, 62))

def train_ch6(train_iter,test_iter,num_epochs,lr,device,model_name,index):
    def init_weights(m):
        if type(m) == nn.Linear or type(m)==nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    net = vgg(conv_arch)
    net.apply(init_weights)
    print('train on', device)
    net.to(device)
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
        out_str = 'cross_{}_epoch:{}/{}\t'.format(index,epoch,num_epochs)
        # print('epoch:{}'.format(epoch),end='\t')
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
        out_str+=f'loss {train_l:.3f},train acc {train_acc:.3f},'f'test acc {test_acc:.3f}'
        print(out_str)
        with open('log/log_{}_{}.log'.format(model_name,index),'a',encoding='utf-8') as wf:
            wf.write(out_str+'\n')
    
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
        img = img.convert('RGB')  
        # img = img.convert('L')
        resize = transforms.Resize((224, 224))  
        img = resize(img)  
        to_tensor = transforms.ToTensor()
        img = to_tensor(img) 
        if self.data_type == 'test':
            y = sample[1]
        else:
            y = self.classes.index(sample[1])
        return img,y
    
def test(test_loader,weights,device,model_name,index):
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    net = vgg(conv_arch)
    net.load_state_dict(torch.load(weights,weights_only=True))
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
    with open('result/{}_cross_{}_{}.csv'.format(model_name,index,datetime.now().strftime("%Y-%m-%d_%H:%M:%S")),'w',encoding='utf-8') as wf:
        wf.write('ImageID,label\n')
        wf.write('\n'.join(result))

if __name__ == '__main__':
    datasets_root = Path('../data/traffic_sign/')
    data_root = Path('./data/traffic_sign/split_1/')
    dataset(datasets_root)
    batch_size = 112
    model_name = 'VGGv1-0002'
    parser = argparse.ArgumentParser(description='')  
    parser.add_argument('--index', type=str, help='交叉验证的划分号',default=0)
    args = parser.parse_args()
    cross_index = int(args.index)
    train_loader = DataLoader(dataset=TrafficSign(datasets_root,data_root,'train'),batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(dataset=TrafficSign(datasets_root,data_root,'val'),batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(dataset=TrafficSign(datasets_root,data_root,'test'),batch_size=batch_size)
    lr, num_epochs = 0.05, 25
    train_ch6( train_loader, val_loader, num_epochs, lr, d2l.try_gpu(), model_name, cross_index)
    test(test_loader,'weights/{}_{}_best_weights.pth'.format(model_name,cross_index),d2l.try_gpu(), model_name,cross_index)
