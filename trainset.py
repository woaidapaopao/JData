# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import datetime as dt

def getlabel(label_start,label_end):
   # label_start = dt.datetime.strptime(label_start,"%Y-%m-%d")
   # label_end = dt.datetime.strptime(label_end,"%Y-%m-%d")#标签结束时间
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
    
    train_end = dt.datetime.strptime(train_end1,"%Y-%m-%d")
    train_start = dt.datetime.strptime(train_start1,"%Y-%m-%d")#训练数据开始时间
    
    label_start = dt.datetime.strptime(label_start1,"%Y-%m-%d")
    label_end = dt.datetime.strptime(label_end1,"%Y-%m-%d")#标签结束时间

    userf = pd.read_csv('./feature/user_feature%s_%s.csv' % (train_start,train_end))
    prof = pd.read_csv('./feature/product_feature%s_%s.csv' % (train_start, train_end))
    uif = pd.read_csv('./feature/user_product_feature%s_%s.csv' % (train_start, train_end))
    ucf = pd.read_csv('./feature/user_cate_feature%s_%s.csv' % (train_start, train_end))
    trainact = pd.merge(uif, userf, how='left', on='user_id')
    trainact = pd.merge(trainact,prof, how='left', on='sku_id')
    trainact = pd.merge(trainact,ucf, how='left', on='user_id')
    label = getlabel(label_start,label_end)
    trainact = pd.merge(trainact,label, how='left', on=['user_id','sku_id'])
    #trainact.fillna(0)
    '''
    ui = trainact[['user_id', 'sku_id']].copy()
    label = trainact['label'].copy()
    del trainact['user_id']
    del trainact['sku_id']
    del trainact['label']
    '''
    #label_end = dt.datetime.strptime(label_end,"%Y-%m-%d")#标签结束时间
    trainact.to_csv('./uncleanData/traindata%s_%s.csv' % (train_start1, label_end1), index = None)

def test_set(train_start1,train_end1):
    train_start = dt.datetime.strptime(train_start1,"%Y-%m-%d")#测试数据开始
    train_end = dt.datetime.strptime(train_end1,"%Y-%m-%d")#测试数据结束
    userf = pd.read_csv('./feature/user_feature%s_%s.csv' % (train_start, train_end))
    prof = pd.read_csv('./feature/product_feature%s_%s.csv' % (train_start, train_end))
    uif = pd.read_csv('./feature/user_product_feature%s_%s.csv' % (train_start, train_end))
    ucf = pd.read_csv('./feature/user_cate_feature%s_%s.csv' % (train_start, train_end))
    trainact = pd.merge(uif, userf, how='left', on='user_id')
    trainact = pd.merge(trainact,prof, how='left', on='sku_id')
    trainact = pd.merge(trainact,ucf, how='left', on='user_id')
    #trainact.fillna(0)
    '''
    ui = trainact[['user_id', 'sku_id']].copy()
    label = trainact['label'].copy()
    del trainact['user_id']
    del trainact['sku_id']
    del trainact['label']
    '''
    trainact.to_csv('./uncleanData/testdata%s_%s.csv' % (train_start1, train_end1), index = None)
