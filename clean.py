# -*- coding:utf-8 -*-
#调用trainset.py中的函数，传入字符串类型的时间
#其中cleanTrainData传入训练开始，训练结束，标签结束三个时间时间
#cleanTestData用于产生最终的提交数据的测试集
import pandas as pd
import numpy as np
import datetime as dt
from trainset import train_set
from trainset import test_set
from trainset import getlabel
import os

def cleanTrainData(train_start, train_end,label_start, label_end):
    #读取未清理数据
    dump_path = './uncleanData/traindata%s_%s.csv' % (train_start, label_end)
    if os.path.exists(dump_path):
        train = pd.read_csv(dump_path, index=None)
    else:
        train_set(train_start, train_end, label_start, label_end)
        train = pd.read_csv(dump_path, index=None)
    #清理方法
    #这里的数据暂时只简单清理一下
    '''
    #这里注销的部分人为处理太严重了就暂时不用了
    train_start1 = dt.datetime.strptime(train_start,"%Y-%m-%d")
    train_end1 = dt.datetime.strptime(train_end_end ,"%Y-%m-%d")#标签结束时间
    userf = pd.read_csv('./feature/user_feature%s_%s.csv' % (train_start1, train_end1))
    #去除点击数浏览数很高却没有过购买记录的用户
    userdl = userf[(userf['buy_num'] == 0) & ((userf['click_num'] > 500) | (userf['browse_num'] > 500))].index
    userf = userf.drop(userdl,axis = 0)
    #去除购买数不为0但是却没有过点击数和浏览数，相当于无效数据
    userdl = userf[(userf['buy_num'] != 0) & (userf['browse_num'] == 0) & (userf['click_num'] == 0)].index
    userf = userf.drop(userdl,axis = 0)
    #去除注册天数大于40天，但是近10天没有过行为记录的用户
    userdl = userf[(userf['reg_time'] > 40) & (userf['weight_10day'] == 0)].index
    userf = userf.drop(userdl,axis = 0)
    user_used = userf['user_id']#得到可用的名单
    train = train[train['user_id'].isin(user_used)]
    '''
    #处理NAN,用最大的时间代替，这个不用了还是用nan值吧，因为其它的特征也有nan的情况
    #但是时间特征应该用最大时间代替
    # 商品特征中comment数量、has_bad_comment和'bad_comment_rate'
    # 用户特征中年龄、性别、注册时间、用户等级特征可能有nan
    #这些特征可以在特征提取阶段就填补上，也可以在clean部分补上
    label = train['label'].fillna(0)
    #train = train.fillna(train.max())#这个暂时就用nan
    train['label'] = label
    train.to_csv('./cleaned/trainCleaned%s_%s.csv' % (train_start, label_end), index=None)

def cleanTestData(test_start, test_end):
    #对测试集进行同样的处理
    dump_path = './uncleanData/testdata%s_%s.csv' % (test_start, test_end)
    if os.path.exists(dump_path):
        test = pd.read_csv(dump_path, index=None)
    else:
        test_set(test_start, test_end)
        test = pd.read_csv(dump_path, index=None)
    #清理方法
    '''
    test_start1 = dt.datetime.strptime(test_start,"%Y-%m-%d")
    test_end1 = dt.datetime.strptime(test_end_end ,"%Y-%m-%d")#标签结束时间
    userf = pd.read_csv('./feature/user_feature%s_%s.csv' % (test_start1, test_end1))
    #去除点击数浏览数很高却没有过购买记录的用户
    userdl = userf[(userf['buy_num'] == 0) & ((userf['click_num'] > 500) | (userf['browse_num'] > 500))].index
    userf = userf.drop(userdl,axis = 0)
    #去除购买数不为0但是却没有过点击数和浏览数，相当于无效数据
    userdl = userf[(userf['buy_num'] != 0) & (userf['browse_num'] == 0) & (userf['click_num'] == 0)].index
    userf = userf.drop(userdl,axis = 0)
    #去除注册天数大于40天，但是近10天没有过行为记录的用户
    userdl = userf[(userf['reg_time'] > 40) & (userf['weight_10day'] == 0)].index
    userf = userf.drop(userdl,axis = 0)
    user_used = userf['user_id']#得到可用的名单
    test = test[test['user_id'].isin(user_used)]
    #处理NAN,用最大的时间代替
    '''
    #test = test.fillna(test.max())
    test.to_csv('./cleaned/testCleaned%s_%s.csv' % (test_start, test_end), index=None)


