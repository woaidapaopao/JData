# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from collections import Counter
import datetime as dt
#分为0201-0320,0215-0405,0225-0415
START_DATE = dt.datetime.strptime('2016-2-20', "%Y-%m-%d")#当前时间
END_DATE = dt.datetime.strptime('2016-4-10', "%Y-%m-%d")#当前时间
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
    #计算转换率
    grouped['delcart_addcart_ratio'] = (np.log(1 + grouped['delcart_num']) - np.log(1 + grouped['addcart_num'])).map(lambda x: '%.2f' % x)
    grouped['buy_addcart_ratio'] = (np.log(1 + grouped['buy_num']) - np.log(1 + grouped['addcart_num'])).map(lambda x: '%.2f' % x)
    grouped['buy_browse_ratio'] = (np.log(1 + grouped['buy_num']) - np.log(1 + grouped['browse_num'])).map(lambda x: '%.2f' % x)
    grouped['buy_click_ratio'] = (np.log(1 + grouped['buy_num']) - np.log(1 + grouped['click_num'])).map(lambda x: '%.2f' % x)
    grouped['buy_favor_ratio'] = (np.log(1 + grouped['buy_num']) - np.log(1 + grouped['favor_num'])).map(lambda x: '%.2f' % x)
    '''
    grouped['delcart_addcart_ratio'] = (grouped['delcart_num'] / grouped['addcart_num']) if(grouped['delcart_num'] / grouped['addcart_num'] < 1.) else 1.
    grouped['buy_addcart_ratio'] = (grouped['buy_num'] / grouped['addcart_num']) if (grouped['buy_num'] / grouped['addcart_num'] < 1.).all() else 1.
    grouped['buy_browse_ratio'] = (grouped['buy_num'] / grouped['browse_num']) if (grouped['buy_num'] / grouped['browse_num'] < 1.).all() else 1.
    grouped['buy_click_ratio'] = (grouped['buy_num'] / grouped['click_num']) if (grouped['buy_num'] / grouped['click_num'] < 1.).all() else 1.
    grouped['buy_favor_ratio'] = (grouped['buy_num'] / grouped['favor_num']) if (grouped['buy_num'] / grouped['favor_num'] < 1.).all() else 1.
    '''
    #下面是活跃度特征
    timeused = (END_DATE - grouped['time']).map(lambda x: x.days)
    for i in {0, 1, 3, 7, 14, 20}:
        us = grouped[timeused <= i]
        type_cnt = Counter(us['type'])
        grouped['weight_%sday' % i] = 0.015 * type_cnt[1] + 0.6 * type_cnt[2] + 0.3 * type_cnt[4] + \
            1.0 * type_cnt[4] + 0.2*type_cnt[5] + 0.01 * type_cnt[6]
    del grouped['time']
    del grouped['type']
    return grouped
'''
    这里是最关键的地方，可用的还有：
    1、用户最近浏览、点击、收藏、加购物车、购买时间(完成)(暂时抛弃)
    2、用户点击、收藏、加购物车、购买量(完成)
    3、用户转化率即用户购买量分别除以用户点击、收藏、加购物车这三类行为数（完成）
    4、平均加购物车（收藏、点击、浏览）到购买的时间差
    5、用户最近3天、7天、10天的浏览、点击、收藏、加购物车、购买加权值(完成)
    6、删除购物车加购物车的比（完成）
'''


def merge_action_data():  #合并用户数据,这里子集和全集都要提取
    user = pd.read_csv(TRAIN_FILE)  #读取训练天数的数据
    user['time'] = pd.to_datetime(user['time'])
    user = user[(user['time'] >= START_DATE) & (user['time'] <= END_DATE)]
    grouped = user[['user_id', 'type', 'time']].groupby('user_id').apply(add_type_count)#计数特征提取
    grouped = grouped.drop_duplicates('user_id')
    return grouped


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


def tranform_user_regtime(df):
    if (df >= 0) & (df < 10):
        df = 0
    elif (df >= 10) & (df < 30):
        df = 1
    elif (df >= 30) & (df < 60):
        df = 2
    elif (df >= 60) & (df < 120):
        df = 3
    elif (df >= 120) & (df < 360):
        df = 4
    elif (df >= 360):
        df = 5
    else:
        df = -1
    return df



#对nan进行预处理
def process_user_feat(user_base):
    #处理注册时间
    user_base['user_reg_tm'] = user_base['user_reg_tm'].map(tranform_user_regtime)
    reg_time = pd.get_dummies(user_base['user_reg_tm'], prefix="reg_time")
    del user_base['user_reg_tm']
    #处理年龄
    user_base['age'] = user_base['age'].map(tranform_user_age)
    age_df = pd.get_dummies(user_base['age'], prefix='age')
    del user_base['age']
    #处理性别
    user_base['sex'] = user_base['sex'].fillna(2)#2表示保密
    sex_df = pd.get_dummies(user_base['sex'], prefix='sex')
    del user_base['sex']
    #处理用户等级
    user_base['user_lv_cd'] = user_base['user_lv_cd'].fillna(-1)#-1表示未知
    user_lv_df = pd.get_dummies(user_base['user_lv_cd'], prefix='user_lv_cd')
    del user_base['user_lv_cd']
    user = pd.concat([age_df, sex_df, user_lv_df, reg_time], axis=1)
    user_base = pd.concat([user_base, user], axis=1)
    return user_base


if __name__ == "__main__":
    user_base = pd.read_csv('JData_User.csv', encoding='gbk')#读取用户
    user_base['user_reg_tm'] = pd.to_datetime(user_base['user_reg_tm'])
    user_base = user_base[user_base['user_reg_tm'] <= END_DATE]
    user_base['user_reg_tm'] = (END_DATE - user_base['user_reg_tm']).map(lambda x: x.days)#转换为注册天数
    
    user_behavior = merge_action_data()#提取用户行为特征
    user_behavior = pd.merge(user_behavior, user_base, on=['user_id'], how='left')#这里就会有NAN值产生
    user_behavior = process_user_feat(user_behavior)#处理数据
    user_behavior.to_csv('./feature/user_feature%s_%s.csv' % (START_DATE, END_DATE), index=None)
