# -*- coding:utf-8 -*-
#用户商品对特征
#这里为了方便文件名都是用datatime的方式写的
import pandas as pd
import numpy as np
import datetime as dt
from collections import Counter
import math

START_DATE = dt.datetime.strptime('2016-2-01', "%Y-%m-%d")#开始时间
END_DATE = dt.datetime.strptime('2016-4-15', "%Y-%m-%d")#当前时间
TRAIN_FILE = 'TrainDataAll.csv'



def user_cate_num(grouped):
    #数量特征，相当于历史交互特征
    timeused = (END_DATE - grouped['time']).map(lambda x: x.days)#这里将所有的时间到截止日期的时间都计算了出来
    #全部ui历史交互
    type_cnt = Counter(grouped['type'])
    grouped['uc_browse_num'] = type_cnt[1]
    grouped['uc_addcart_num'] = type_cnt[2]
    grouped['uc_delcart_num'] = type_cnt[3]
    grouped['uc_buy_num'] = type_cnt[4]
    grouped['uc_favor_num'] = type_cnt[5]
    grouped['uc_click_num'] = type_cnt[6]
    #近期交互
    for i in {1, 3, 5, 7, 14, 20}:
        #计数特征
        us = grouped[timeused <= i]
        type_cnt = Counter(us['type'])
        grouped['uc_browse_num%s' % i] = type_cnt[1]
        grouped['uc_addcart_num%s' % i] = type_cnt[2]
        grouped['uc_delcart_num%s' % i] = type_cnt[3]
        grouped['uc_buy_num%s' % i] = type_cnt[4]
        grouped['uc_favor_num%s' % i] = type_cnt[5]
        grouped['uc_click_num%s' % i] = type_cnt[6]
        #近期加权行为特征
        grouped['uc_weight_%sday' % i] = 0.015 * type_cnt[1] + 0.6 * type_cnt[2] + 0.3 * type_cnt[4] + \
            1.0 * type_cnt[4] + 0.2*type_cnt[5] + 0.01 * type_cnt[6]
        #用户衰减加权行为特征（没提取）

    groupedre = grouped.drop('type', axis=1)
    groupedre = groupedre.drop('time', axis=1)
    return groupedre


def user_pro_num(grouped):
    #数量特征，相当于历史交互特征
    timeused = (END_DATE - grouped['time']).map(lambda x: x.days)#这里将所有的时间到截止日期的时间都计算了出来
    #全部ui历史交互
    type_cnt = Counter(grouped['type'])
    grouped['ui_browse_num'] = type_cnt[1]
    grouped['ui_addcart_num'] = type_cnt[2]
    grouped['ui_delcart_num'] = type_cnt[3]
    grouped['ui_buy_num'] = type_cnt[4]
    grouped['ui_favor_num'] = type_cnt[5]
    grouped['ui_click_num'] = type_cnt[6]
    #近期交互
    for i in {1, 3, 5, 7, 14, 20}:
        #计数特征
        us = grouped[timeused <= i]
        type_cnt = Counter(us['type'])
        grouped['ui_browse_num%s' % i] = type_cnt[1]
        '''
        grouped['ui_addcart_num%s' % i] = type_cnt[2]#其实这几个特征有点问题因为不可能同一商品加了几次购物车
        grouped['ui_delcart_num%s' % i] = type_cnt[3]
        grouped['ui_buy_num%s' % i] = type_cnt[4]
        grouped['ui_favor_num%s' % i] = type_cnt[5]
        '''
        grouped['ui_click_num%s' % i] = type_cnt[6]
        #近期加权行为特征
        grouped['ui_weight_%sday' % i] = 0.015 * type_cnt[1] + 0.6 * type_cnt[2] + 0.3 * type_cnt[4] + \
            1.0 * type_cnt[4] + 0.2*type_cnt[5] + 0.01 * type_cnt[6]
        #用户衰减加权行为特征（没提取）
    groupedre = grouped.drop('type', axis=1)
    groupedre = groupedre.drop('time', axis=1)
    return groupedre

def user_pro_cate_feature(df):
    #比例特征（转换为排序特征）
    for i in {1, 3, 5, 7, 14, 20}:
        df['uic_browse_radio%s' % i] = float('%.2f' % (math.log(1 + df['ui_browse_num%s' % i] / df['uc_browse_num%s' % i])))
        df['uic_click_radio%s' % i] = float('%.2f' % (math.log(1 + df['ui_click_num%s' % i] / df['uc_click_num%s' % i])))
        df['uic_weight_radio_%sday' % i] = float('%.2f' % (math.log(1 + df['ui_weight_%sday' % i] / df['uc_weight_%sday' % i])))
    return df
if __name__ == "__main__":
    train = pd.read_csv(TRAIN_FILE)
    train['time'] = pd.to_datetime(train['time'])
    train = train[(train['time'] >= START_DATE) & (train['time'] <= END_DATE)]
    #每次只使用5天之内且商品类别为8的ui对
    start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=5)#5是可调数据
    user_act = train[(train['time'] <= END_DATE) & (train['time'] > start_days) & (train['cate'] == 8)][['user_id', 'sku_id']]
    user_act = user_act.drop_duplicates()
    user_act = pd.merge(train, user_act, on=['user_id', 'sku_id'], how='right')#这样就得到了用户商品对

    #计算用户类别uc特征
    user_cate = user_act[['user_id', 'type', 'time']].groupby(['user_id']).apply(user_cate_num)#提取用户类别对特征
    user_cate = user_cate.drop_duplicates()
    #计算用户商品ui特征
    user_product = user_act[['user_id', 'sku_id', 'type', 'time']].groupby(['user_id', 'sku_id']).apply(user_pro_num)#提取用户商品对特征
    user_product = user_product.drop_duplicates()

    user_i_p_c = pd.merge(user_product, user_cate, on=['user_id'], how='right')#这样就得到了用户商品对
    #计算ui&uc联合特征
    user_i_p_c = user_i_p_c.groupby('user_id').apply(user_pro_cate_feature)

    grouped.to_csv('./feature/user_product_cate_feature%s_%s.csv' % (START_DATE, END_DATE), index=None)