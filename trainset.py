# -*- coding:utf-8 -*-
#这个文件不用直接调用，可以在clean中调用
import pandas as pd
import numpy as np
import datetime as dt

def getlabel(label_start,label_end):
    #提取标签信息
    TRAIN_FILE = 'TrainDataAll.csv'
    train = pd.read_csv(TRAIN_FILE)
    train['time'] = pd.to_datetime(train['time'])
    train = train[(train['time'] >= label_start) & (train['time'] <= label_end)]
    train = train[train['type'] == 4]
    train = train[['user_id', 'sku_id']]
    train.drop_duplicates()
    train['label'] = 1
    return train
def train_set(train_start1,train_end1,label_start1,label_end1):
    
    train_end = dt.datetime.strptime(train_end1, "%Y-%m-%d")
    train_start = dt.datetime.strptime(train_start1, "%Y-%m-%d")#训练数据开始时间
    
    label_start = dt.datetime.strptime(label_start1, "%Y-%m-%d")
    label_end = dt.datetime.strptime(label_end1, "%Y-%m-%d")#标签结束时间

    userf = pd.read_csv('./feature/user_feature%s_%s.csv' % (train_start, train_end))
    prof = pd.read_csv('./feature/product_feature%s_%s.csv' % (train_start, train_end))
    uicf = pd.read_csv('./feature/user_product_cate_feature%s_%s.csv' % (train_start, train_end))

    trainact = pd.merge(uicf, userf, how='left', on='user_id')
    trainact = pd.merge(trainact, prof, how='left', on='sku_id')
    label = getlabel(label_start, label_end)
    trainact = pd.merge(trainact, label, how='left', on=['user_id', 'sku_id'])#这时的label部分只有1，其余为nan，这个在clean中处理
    trainact.to_csv('./uncleanData/traindata%s_%s.csv' % (train_start1, label_end1), index=None)

def test_set(test_start1,test_end1):
    test_start = dt.datetime.strptime(test_start1, "%Y-%m-%d")#测试数据开始
    test_end = dt.datetime.strptime(test_end1, "%Y-%m-%d")#测试数据结束
    userf = pd.read_csv('./feature/user_feature%s_%s.csv' % (test_start, test_end))
    prof = pd.read_csv('./feature/product_feature%s_%s.csv' % (test_start, test_end))
    uicf = pd.read_csv('./feature/user_product_cate_feature%s_%s.csv' % (test_start, test_end))
    testact = pd.merge(uicf, userf, how='left', on='user_id')
    testact = pd.merge(testact, prof, how='left', on='sku_id')
    testact.to_csv('./uncleanData/testdata%s_%s.csv' % (test_start1, test_end1), index=None)
