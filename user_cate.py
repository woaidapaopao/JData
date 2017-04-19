# -*- coding:utf-8 -*-
#用户类别对特征
import pandas as pd
import numpy as np
import datetime as dt
from collections import Counter
START_DATE = dt.datetime.strptime('2016-2-05', "%Y-%m-%d")#当前时间
END_DATE = dt.datetime.strptime('2016-4-10', "%Y-%m-%d")#当前时间
TRAIN_FILE = 'TrainDataAll.csv'

def user_pro_num(grouped):
    #数量特征
    timeused = (END_DATE - grouped['time']).map(lambda x: x.days)
    for i in {1, 3, 5, 7, 14, 20}:
        us = grouped[timeused <= i]
        type_cnt = Counter(us['type'])
        grouped['uc_browse_num%s' % i] = type_cnt[1]
        grouped['uc_addcart_num%s' % i] = type_cnt[2]
        grouped['uc_delcart_num%s' % i] = type_cnt[3]
        grouped['uc_buy_num%s' % i] = type_cnt[4]
        grouped['uc_favor_num%s' % i] = type_cnt[5]
        grouped['uc_click_num%s' % i] = type_cnt[6]
    #全部数量
    type_cnt = Counter(grouped['type'])
    grouped['uc_browse_num'] = type_cnt[1]
    grouped['uc_addcart_num'] = type_cnt[2]
    grouped['uc_delcart_num'] = type_cnt[3]
    grouped['uc_buy_num'] = type_cnt[4]
    grouped['uc_favor_num'] = type_cnt[5]
    grouped['uc_click_num'] = type_cnt[6]
    #时间特征
    if type_cnt[1] != 0:
        grouped['uc_latest_browse_time'] = (END_DATE - grouped[grouped['type'] == 1]['time'].max()).days
    else:
        grouped['uc_latest_browse_time'] = np.nan
    if type_cnt[2] != 0:
        grouped['uc_latest_addcart_time'] = (END_DATE - grouped[grouped['type'] == 2]['time'].max()).days
    else:
        grouped['uc_latest_addcart_time'] = np.nan
    if type_cnt[3] != 0:
        grouped['uc_latest_delcart_time'] = (END_DATE - grouped[grouped['type'] == 3]['time'].max()).days
    else:
        grouped['uc_latest_delcart_time'] = np.nan
    if type_cnt[4] != 0:
        grouped['uc_latest_buy_time'] = (END_DATE - grouped[grouped['type'] == 4]['time'].max()).days
    else:
        grouped['uc_latest_buy_time'] = np.nan
    if type_cnt[5] != 0:
        grouped['uc_latest_favor_time'] = (END_DATE - grouped[grouped['type'] == 5]['time'].max()).days
    else:
        grouped['uc_latest_favor_time'] = np.nan
    if type_cnt[6] != 0:
        grouped['uc_latest_click_time'] = (END_DATE - grouped[grouped['type'] == 6]['time'].max()).days
    else:
        grouped['uc_latest_click_time'] = np.nan
    
    del grouped['type']
    del grouped['time']
    return grouped
'''
1、用户对类别点击、收藏、加购物车、购买的最近时间。
2、用户对类别点击、收藏、加购物车、购买的次数
'''
if __name__ == "__main__":
    train = pd.read_csv(TRAIN_FILE)
    train['time'] = pd.to_datetime(train['time'])
    train = train[(train['time'] >= START_DATE) & (train['time'] <= END_DATE)]
    user_pro = train[train['cate'] == 8][['user_id']]#用户类别对
    user_pro = user_pro.drop_duplicates()
    user_pro = pd.merge(train, user_pro, on='user_id', how='right')#这样就得到了用户类别对,由于针对一个商品，其实这个就是近似包含了类别特征
    user_pro['time'] = pd.to_datetime(user_pro['time'])
    grouped = user_pro[['user_id', 'type', 'time']].groupby(['user_id']).apply(user_pro_num)#与ui的区别就在这里
    #这里决定在这一阶段填补nan值，让所有nan值等于时间最长的交互时间
    grouped = grouped.drop_duplicates()
    grouped.to_csv('./feature/user_cate_feature%s_%s.csv' % (START_DATE, END_DATE), index = None)