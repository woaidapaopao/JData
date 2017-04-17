# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from collections import Counter
import datetime as dt
START_DATE = dt.datetime.strptime('2016-2-01', "%Y-%m-%d")#当前时间
END_DATE = dt.datetime.strptime('2016-4-05', "%Y-%m-%d")#当前时间
TRAIN_FILE = 'TrainDataAll.csv'
def add_type_count(grouped):
    #计数特征提取
    type_cnt = Counter(grouped['type'])#也可以用value_counts()，但是返回的是一个Series，如果不存在这个type，会出错
    grouped['browse_num'] = type_cnt[1]
    grouped['addcart_num'] = type_cnt[2]
    grouped['delcart_num'] = type_cnt[3]
    grouped['buy_num'] = type_cnt[4]
    grouped['favor_num'] = type_cnt[5]
    grouped['click_num'] = type_cnt[6]
    #计算用户最近一次的操作距离现在的时间
    if type_cnt[1] != 0:
        grouped['latest_browse_time'] = (END_DATE - grouped[grouped['type'] == 1]['time'].max()).days
    else:
        grouped['latest_browse_time'] = np.nan
    if type_cnt[2] != 0:
        grouped['latest_addcart_time'] = (END_DATE - grouped[grouped['type'] == 2]['time'].max()).days
    else:
        grouped['latest_addcart_time'] = np.nan
    if type_cnt[3] != 0:
        grouped['latest_delcart_time'] = (END_DATE - grouped[grouped['type'] == 3]['time'].max()).days
    else:
        grouped['latest_delcart_time'] = np.nan
    if type_cnt[4] != 0:
        grouped['latest_buy_time'] = (END_DATE - grouped[grouped['type'] == 4]['time'].max()).days
    else:
        grouped['latest_buy_time'] = np.nan
    if type_cnt[5] != 0:
        grouped['latest_favor_time'] = (END_DATE - grouped[grouped['type'] == 5]['time'].max()).days
    else:
        grouped['latest_favor_time'] = np.nan
    if type_cnt[6] != 0:
        grouped['latest_click_time'] = (END_DATE - grouped[grouped['type'] == 6]['time'].max()).days
    else:
        grouped['latest_click_time'] = np.nan
    
    #计算转换率
    grouped['delcart_addcart_ratio'] = (grouped['delcart_num'] / grouped['addcart_num']) if(grouped['delcart_num'] / grouped['addcart_num'] < 1.) else 1.
    grouped['buy_addcart_ratio'] = (grouped['buy_num'] / grouped['addcart_num']) if (grouped['buy_num'] / grouped['addcart_num'] < 1.).all() else 1.
    grouped['buy_browse_ratio'] = (grouped['buy_num'] / grouped['browse_num']) if (grouped['buy_num'] / grouped['browse_num'] < 1.).all() else 1.
    grouped['buy_click_ratio'] = (grouped['buy_num'] / grouped['click_num']) if (grouped['buy_num'] / grouped['click_num'] < 1.).all() else 1.
    grouped['buy_favor_ratio'] = (grouped['buy_num'] / grouped['favor_num']) if (grouped['buy_num'] / grouped['favor_num'] < 1.).all() else 1.
    #最近3天的加权值，到时候直接改成上面的那种
    #这样计算加权值可以得到最近的活跃度
  #  for i in {1,3,5,7,14,20,30}:
    timeused = (END_DATE - grouped['time']).map(lambda x: x.days)
    type_cnt2 = Counter(grouped[timeused <= 3]['type'])
    '''
    grouped['weight_3day'] = grouped['buy_browse_ratio']*type_cnt2[1] + grouped['buy_addcart_ratio'] * type_cnt2[2] + \
        1*type_cnt2[4] + grouped['buy_favor_ratio']*type_cnt2[5] + grouped['buy_click_ratio']* type_cnt2[6]
    '''
    grouped['weight_3day'] = 0.02 * type_cnt2[1] + 0.6 * type_cnt2[2] + 0.3 * type_cnt2[4] + \
            1.0 * type_cnt2[4] + 0.2*type_cnt2[5] + 0.01 * type_cnt2[6]
    #最近7天的加权值
    type_cnt2 = Counter(grouped[timeused <= 7]['type'])
    grouped['weight_7day'] = 0.02 * type_cnt2[1] + 0.6 * type_cnt2[2] + 0.3 * type_cnt2[4] + \
            1.0 * type_cnt2[4] + 0.2*type_cnt2[5] + 0.01 * type_cnt2[6]
    type_cnt2 = Counter(grouped[timeused <= 10]['type'])
    grouped['weight_10day'] = 0.02 * type_cnt2[1] + 0.6 * type_cnt2[2] + 0.3 * type_cnt2[4] + \
            1.0 * type_cnt2[4] + 0.2*type_cnt2[5] + 0.01 * type_cnt2[6]
    del grouped['time']
    del grouped['type']
    return grouped
'''
    这里是最关键的地方，可用的还有：
    1、用户最近浏览、点击、收藏、加购物车、购买时间(完成)
    2、用户点击、收藏、加购物车、购买量(完成)
    3、用户转化率即用户购买量分别除以用户点击、收藏、加购物车这三类行为数（完成）
    4、平均加购物车（收藏、点击、浏览）到购买的时间差
    5、用户最近3天、7天、10天的浏览、点击、收藏、加购物车、购买加权值(完成)
    6、删除购物车加购物车的比（完成）
'''
def merge_action_data():#合并用户数据,这里子集和全集都要提取
    user = pd.read_csv(TRAIN_FILE)#读取训练天数的数据
    user['time'] = pd.to_datetime(user['time'])
    user = user[(user['time'] >= START_DATE) & (user['time'] <= END_DATE)]
    grouped = user[['user_id','type','time']].groupby('user_id').apply(add_type_count)#计数特征提取
    grouped = grouped.drop_duplicates('user_id')
    return grouped
'''
    其实这样写很好，因为和直接计算注册时间是等价的，所有人都用一个基准值做基础
    df['user_reg_dt'] = pd.to_datetime(df['user_reg_dt'])
    min_date = min(df['user_reg_dt'])
    df['user_reg_diff'] = [int(i.days) for i in (df['user_reg_dt'] - min_date)]
'''
def tranform_user_age(df):
    if df == u"15岁以下":
        df = 0
    elif df == u"16-25岁":
        df = 1
    elif df == u"26-35岁":
        df = 2
    elif df == u"36-45岁":
        df = 3
    elif df == u"46-55岁":
        df = 4
    elif df == u"56岁以上":
        df = 5
    else:
        df = -1
    return df
if __name__ == "__main__":
    user_base = pd.read_csv('JData_User.csv',encoding = 'gbk')#读取用户
    user_base['user_reg_tm'] = pd.to_datetime(user_base['user_reg_tm'])#注册时间表
    user_base = user_base[user_base['user_reg_tm'] <= END_DATE]
    user_base['reg_time'] = (END_DATE - user_base['user_reg_tm']).map(lambda x: x.days)
    del user_base['user_reg_tm']
    age = user_base['age'].map(tranform_user_age)
    user_base['age'] = age

    user_behavior = merge_action_data()
    user_behavior = pd.merge(user_behavior,user_base, on = ['user_id'], how = 'left')
    user_behavior.to_csv('./feature/user_feature%s_%s.csv' % (START_DATE,END_DATE), index = None)