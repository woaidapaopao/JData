# -*- coding:utf-8 -*-
#用户商品对特征
#这里为了方便文件名都是用datatime的方式写的
import pandas as pd
import numpy as np
import datetime as dt
from collections import Counter
#这一部分需要计算很多截止时间的特征如2016-2-01到2016-3-01、2016-3-06到2016-04-06
#2016-03-15到2016-04-15
START_DATE = dt.datetime.strptime('2016-2-01', "%Y-%m-%d")#开始时间
END_DATE = dt.datetime.strptime('2016-4-15', "%Y-%m-%d")#当前时间
TRAIN_FILE = 'TrainDataAll.csv'
def user_pro_num(grouped):
    #数量特征，相当于历史交互特征
    timeused = (END_DATE - grouped['time']).map(lambda x: x.days)#这里将所有的时间到截止日期的时间都计算了出来
    for i in {1, 3, 5, 7, 14, 20}:
        us = grouped[timeused <= i]
        type_cnt = Counter(us['type'])
        grouped['ui_browse_num%s' % i] = type_cnt[1]
        grouped['ui_addcart_num%s' % i] = type_cnt[2]
        grouped['ui_delcart_num%s' % i] = type_cnt[3]
        grouped['ui_buy_num%s' % i] = type_cnt[4]
        grouped['ui_favor_num%s' % i] = type_cnt[5]
        grouped['ui_click_num%s' % i] = type_cnt[6]
    #全部数量
    type_cnt = Counter(grouped['type'])
    grouped['ui_browse_num'] = type_cnt[1]
    grouped['ui_addcart_num'] = type_cnt[2]
    grouped['ui_delcart_num'] = type_cnt[3]
    grouped['ui_buy_num'] = type_cnt[4]
    grouped['ui_favor_num'] = type_cnt[5]
    grouped['ui_click_num'] = type_cnt[6]
    #时间特征
    grouped['ui_latest_time'] = timeused.max()#计算最近的交互时间，方便选择用户商品对范围
    if type_cnt[1] != 0:
        grouped['ui_latest_browse_time'] = (END_DATE - grouped[grouped['type'] == 1]['time'].max()).days
    else:
        grouped['ui_latest_browse_time'] = np.nan
    if type_cnt[2] != 0:
        grouped['ui_latest_addcart_time'] = (END_DATE - grouped[grouped['type'] == 2]['time'].max()).days
    else:
        grouped['ui_latest_addcart_time'] = np.nan
    if type_cnt[3] != 0:
        grouped['ui_latest_delcart_time'] = (END_DATE - grouped[grouped['type'] == 3]['time'].max()).days
    else:
        grouped['ui_latest_delcart_time'] = np.nan
    if type_cnt[4] != 0:
        grouped['ui_latest_buy_time'] = (END_DATE - grouped[grouped['type'] == 4]['time'].max()).days
    else:
        grouped['ui_latest_buy_time'] = np.nan
    if type_cnt[5] != 0:
        grouped['ui_latest_favor_time'] = (END_DATE - grouped[grouped['type'] == 5]['time'].max()).days
    else:
        grouped['ui_latest_favor_time'] = np.nan
    if type_cnt[6] != 0:
        grouped['ui_latest_click_time'] = (END_DATE - grouped[grouped['type'] == 6]['time'].max()).days
    else:
        grouped['ui_latest_click_time'] = np.nan
    del grouped['type']
    del grouped['time']
    return grouped
'''
1、用户对商品点击、收藏、加购物车、购买的最近时间。
2、用户对商品点击、收藏、加购物车、购买的次数1,3,5,7,14,20,30
'''
if __name__ == "__main__":
    train = pd.read_csv(TRAIN_FILE)
    train['time'] = pd.to_datetime(train['time'])
    train = train[(train['time'] >= START_DATE) & (train['time'] <= END_DATE)]
    user_pro = train[train['cate'] == 8][['user_id', 'sku_id']]#用户商品对
    user_pro = user_pro.drop_duplicates()
    user_pro = pd.merge(train, user_pro, on=['user_id', 'sku_id'], how='right')#这样就得到了用户商品对
    grouped = user_pro[['user_id', 'sku_id', 'type', 'time']].groupby(['user_id', 'sku_id']).apply(user_pro_num)#提取用户商品对特征
    grouped = grouped.drop_duplicates()
    grouped.to_csv('./feature/user_product_feature%s_%s.csv' % (START_DATE, END_DATE), index=None)