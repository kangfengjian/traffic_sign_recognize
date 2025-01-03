import os
import pandas as pd
from collections import defaultdict

def vote():
    result = defaultdict(lambda:[0]*62)
    result_dir = './result/'
    model_name = 'ResNet-B-0001'
    for i in os.listdir(result_dir):
        if i.startswith(model_name):
            df = pd.read_csv(result_dir+i)
            for j in df.index:
                result[df.loc[j,'ImageID']][df.loc[j,'label']]+=1

    for i in result:
        result[i] = result[i].index(max(result[i]))

    df = pd.DataFrame({'ImageID':result.keys(),'label':result.values()})
    df.to_csv('{}_vote_result.csv'.format(model_name),index=False)

def acc():
    std = pd.read_csv('test_std.csv',index_col='ImageID')
    pre = pd.read_csv('aaa.csv',index_col='ImageID')
    n = 0
    for i in std.index:
        if std.loc[i,'label']==pre.loc[i,'label']:
            n+=1
    print('{}/{},{}'.format(n,std.shape[0],n/std.shape[0]))

if __name__ =='__main__':
    vote()

