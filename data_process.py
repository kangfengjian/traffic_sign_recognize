from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split # pip install scikit-learn
from sklearn.model_selection import KFold



'''
将训练集按照1:9划分为训练集和验证集，并将训练集、验证集、测试集以地址形式存储在项目中
当 exist_ok==True 时，若已存在文件，则不重新生成；若不存在则生成
当 exist_ok==False 时，生成
将类别信息存储在classes.txt中
'''

datasets_root = Path('../data/traffic_sign/')
data_root = Path('./data/traffic_sign_B/')
exist_ok=True
data_root.mkdir(parents=True, exist_ok=True)
# 获取类别信息
with open(data_root/'classes.json','r',encoding='utf-8') as rf:
    classes = json.loads(rf.read())
# 获取训练集信息
train_path = datasets_root/Path('train_set')
train_data_x, train_data_y = list(), list()
for item in train_path.rglob('*'):
    if item.is_file():
        train_data_x.append(item.relative_to(datasets_root))
        train_data_y.append(classes[item.parent.name])
# 获取测试集a的信息
test_data_x, test_data_y = list(), list()
test_a_class = pd.read_csv(datasets_root/'test_set/test_A.csv',index_col='ImageID')
test_a_class = {i:test_a_class.loc[i,'label']   for i in test_a_class.index}
test_path = datasets_root/Path('test_set')
for item in test_path.rglob('*'):
    if item.is_file() and item.suffix!='.csv':
        test_data_x.append(item.relative_to(datasets_root))
        test_data_y.append(test_a_class[item.name]) 
# 两个数据集合并
data_x = train_data_x+test_data_x
data_y = train_data_y+test_data_y
print(len(data_x))
print(len(data_y))
# 交叉验证数据划分
kf = KFold(n_splits=10, shuffle=True, random_state=42)
i = 0
for train_index, val_index in kf.split(data_x):
    with open(data_root/Path('train_{}.csv'.format(i)),'w',encoding='utf-8') as wf:
        wf.write('\n'.join(['{},{}'.format(data_x[j],data_y[j]) for j in train_index]))
    with open(data_root/Path('test_{}.csv'.format(i)),'w',encoding='utf-8') as wf:
        wf.write('\n'.join(['{},{}'.format(data_x[j],data_y[j]) for j in val_index]))
    i+=1
