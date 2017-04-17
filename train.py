# -*- coding:utf-8 -*-
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import numpy as np
import datetime as dt
'''
TRAIN_START1 = dt.datetime.strptime('2016-2-01',"%Y-%m-%d")#训练数据开始时间
TRAIN_END1 = dt.datetime.strptime('2016-4-05',"%Y-%m-%d")
LABEL_START1 = dt.datetime.strptime('2016-4-6',"%Y-%m-%d")
LABEL_END1 = dt.datetime.strptime('2016-4-10',"%Y-%m-%d")#标签结束时间

TRAIN_START2 = dt.datetime.strptime('2016-2-01',"%Y-%m-%d")#训练数据开始时间
TRAIN_END2 = dt.datetime.strptime('2016-4-05',"%Y-%m-%d")
LABEL_START2 = dt.datetime.strptime('2016-4-6',"%Y-%m-%d")
LABEL_END2 = dt.datetime.strptime('2016-4-10',"%Y-%m-%d")#标签结束时间

START_DATE = dt.datetime.strptime('2016-4-11',"%Y-%m-%d")#测试数据开始
END_DATE = dt.datetime.strptime('2016-4-15',"%Y-%m-%d")#测试数据结束
'''
def offlineTest(pred, label):#label为真实值，pred为预测值
    uselabel = label
    result = pred
    #计算所有商品对
    all_user_item_pair = uselabel['user_id'].map(str) + '-' + uselabel['sku_id'].map(str)
    all_user_item_pair = np.array(all_user_item_pair)
    #所有购买用户
    all_user_set = uselabel['user_id'].unique()

    #预测的购买用户和用户商品对
    all_user_test_set = result['user_id'].unique()
    all_user_test_item_pair = result['user_id'].map(str) + '-' + result['sku_id'].map(str)
    all_user_test_item_pair = np.array(all_user_test_item_pair)

    #计算评价标准
    pos, neg = 0,0
    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1
    all_user_pre = 1.0 * pos / (pos + neg)#计算精确度TP/(TP+FP)
    all_user_recall = 1.0 * pos / len(all_user_set)#计算召回率TP/(TP+FN)
    print '所有用户中预测购买用户的精确度为 ' + str(all_user_pre)
    print '所有用户中预测购买用户的召回率' + str(all_user_recall)
    pos, neg = 0, 0
    for user_item_pair in all_user_test_item_pair:
        if user_item_pair in all_user_item_pair:
            pos += 1
        else:
            neg += 1
    all_item_pre = 1.0 * pos / ( pos + neg)
    all_item_recall = 1.0 * pos / len(all_user_item_pair)
    print '所有用户中预测购买商品的精确度为 ' + str(all_item_pre)
    print '所有用户中预测购买商品的召回率' + str(all_item_recall)
    F11 = 6.0 * all_user_recall * all_user_pre / (5.0 * all_user_recall + all_user_pre)
    F12 = 5.0 * all_item_pre * all_item_recall / (2.0 * all_item_recall + 3 * all_item_pre)
    score = 0.4 * F11 + 0.6 * F12
    print 'F11=' + str(F11)
    print 'F12=' + str(F12)
    print 'score=' + str(score)

def subOnline():
    train_start = '2016-02-01'
    label_end = '2016-04-05'
    test_start = '2016-02-01'
    test_end = '2016-04-15'

    train = pd.read_csv('trainCleaned%s_%s.csv' % (train_start, label_end))
    user_index = train[['user_id','sku_id']]
    label = train['label']
    traindata = train.drop(['user_id','sku_id','label'],axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(traindata.values, label.values, test_size=0.2, random_state=0)
    dtrain=xgb.DMatrix(X_train, label=y_train)
    dtest=xgb.DMatrix(X_test, label=y_test)
    param = {'learning_rate' : 0.1, 'n_estimators': 1000, 'max_depth': 3,
        'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
        'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 283
    param['nthread'] = 4
    #param['eval_metric'] = "auc"
    plst = param.items()
    plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst=xgb.train(plst, dtrain, num_round, evallist)

    test = pd.read_csv('testCleaned%s_%s.csv' % (test_start, test_end))
    sub_user_index = test[['user_id','sku_id']]
    sub_trainning_data = test.drop(['user_id','sku_id'],axis = 1)
    sub_trainning_data = xgb.DMatrix(sub_trainning_data.values)
    y = bst.predict(sub_trainning_data)
    sub_user_index['label'] = y
    pred = sub_user_index[sub_user_index['label'] >= 0.03]#这里可以好好调整
    pred = pred[['user_id', 'sku_id']]
    pred = pred.groupby('user_id').first().reset_index()
    pred['user_id'] = pred['user_id'].astype(int)
    pred.to_csv('submission.csv', index=False, index_label=False)

def offlineTesT():
    train_start1 = '2016-02-01'
    label_end1 = '2016-04-05'
    train_start2 = '2016-02-05'
    label_end2 = '2016-04-15'
    train = pd.read_csv('trainCleaned%s_%s.csv' % (train_start1, label_end1))
    user_index = train[['user_id','sku_id']]
    label = train['label']
    traindata = train.drop(['user_id','sku_id','label'])
    X_train, X_test, y_train, y_test = train_test_split(traindata, label, test_size=0.2, random_state=0)
    dtrain=xgb.DMatrix(X_train, label=y_train)
    dtest=xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth': 10, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 4000
    param['nthread'] = 4
    param['eval_metric'] = "auc"
    plst = param.items()
    plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst=xgb.train( plst, dtrain, num_round, evallist)
#
    test = pd.read_csv('trainCleaned%s_%s.csv' % (train_start2, label_end2))
    user_index2 = test[['user_id','sku_id']]
    label2 = test['label']
    testdata = test.drop(['user_id', 'sku_id', 'label'])
    testMa = xgb.DMatrix(testdata.values)
    y = bst.predict(testMa)
    pre = user_index2[y == 1]
    offlineTest(pre,label2)

if __name__ == '__main__':
    #xgboost_cv()
    subOnline()