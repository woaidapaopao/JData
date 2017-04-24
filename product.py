# -*- coding:utf-8 -*-
#商品特征
import pandas as pd
import numpy as np
from collections import Counter
import datetime as dt
START_DATE = dt.datetime.strptime('2016-2-10',"%Y-%m-%d")#当前时间
END_DATE = dt.datetime.strptime('2016-4-01',"%Y-%m-%d")#当前时间
TRAIN_FILE = 'TrainDataAll.csv'


def add_type_count(grouped):
    #计数特征提取
    type_cnt = Counter(grouped['type'])
    grouped['pro_browse_num'] = type_cnt[1]
    grouped['pro_addcart_num'] = type_cnt[2]
    grouped['pro_delcart_num'] = type_cnt[3]
    grouped['pro_buy_num'] = type_cnt[4]
    grouped['pro_favor_num'] = type_cnt[5]
    grouped['pro_click_num'] = type_cnt[6]
    #比率特征，这里设计一个平滑函数用于解决除零问题
    grouped['pro_delcart_addcart_ratio'] = (np.log(1 + grouped['pro_delcart_num']) - np.log(1 + grouped['pro_addcart_num'])).map(lambda x: '%.2f' % x)
    grouped['pro_buy_addcart_ratio'] = (np.log(1 + grouped['pro_buy_num']) - np.log(1 + grouped['pro_addcart_num'])).map(lambda x: '%.2f' % x)
    grouped['pro_buy_browse_ratio'] = (np.log(1 + grouped['pro_buy_num']) - np.log(1 + grouped['pro_browse_num'])).map(lambda x: '%.2f' % x)
    grouped['pro_buy_click_ratio'] = (np.log(1 + grouped['pro_buy_num']) - np.log(1 + grouped['pro_click_num'])).map(lambda x: '%.2f' % x)
    grouped['pro_buy_favor_ratio'] = (np.log(1 + grouped['pro_buy_num']) - np.log(1 + grouped['pro_favor_num'])).map(lambda x: '%.2f' % x)
    '''
    if (grouped['pro_delcart_num'] == 0) & (grouped['pro_addcart_num'] == 0):
        grouped['pro_delcart_addcart_ratio'] = 0.
    elif (grouped['pro_delcart_num']/grouped['pro_addcart_num'] > 1.) | ((grouped['pro_delcart_num'] > 0) & (grouped['pro_addcart_num'] == 0)):
        grouped['pro_delcart_addcart_ratio'] = 1.
    else:
        grouped['pro_delcart_addcart_ratio'] = grouped['pro_delcart_num'] / grouped['pro_addcart_num']
    #买加比
    if (grouped['pro_buy_num'] == 0) & (grouped['pro_addcart_num'] == 0):
        grouped['pro_buy_addcart_ratio'] = 0.
    elif (grouped['pro_buy_num'] / grouped['pro_addcart_num'] > 1.) | ((grouped['pro_buy_num'] > 0) & (grouped['pro_addcart_num'] == 0)):
        grouped['pro_buy_addcart_ratio'] = 1.
    else:
        grouped['pro_buy_addcart_ratio'] = grouped['pro_buy_num'] / grouped['pro_addcart_num']
    #买浏览比
    if (grouped['pro_buy_num'] == 0) & (grouped['pro_browse_num'] == 0):
        grouped['pro_buy_browse_ratio'] = 0.
    elif (grouped['pro_buy_num'] / grouped['pro_browse_num'] > 1.) | ((grouped['pro_buy_num'] > 0) & (grouped['pro_browse_num'] == 0)):
        grouped['pro_buy_browse_ratio'] = 1.
    else:
        grouped['pro_buy_browse_ratio'] = (grouped['pro_buy_num'] / grouped['pro_browse_num'])
    #买点击比
    if (grouped['pro_buy_num'] == 0) & (grouped['pro_click_num'] == 0):
        grouped['pro_buy_click_ratio'] = 0.
    elif (grouped['pro_buy_num'] / grouped['pro_click_num'] > 1.) | ((grouped['pro_buy_num'] > 0) & (grouped['pro_click_num'] == 0)):
        grouped['pro_buy_click_ratio'] = 1.
    else:
        grouped['pro_buy_click_ratio'] = grouped['pro_buy_num'] / grouped['pro_click_num']

    if (grouped['pro_buy_num'] == 0) & (grouped['pro_favor_num'] == 0):
        grouped['pro_buy_favor_ratio'] = 0.
    elif (grouped['pro_buy_num'] / grouped['pro_favor_num'] > 1.) | ((grouped['pro_buy_num'] > 0) & (grouped['pro_favor_num'] == 0)):
        grouped['pro_buy_favor_ratio'] = 1.
    else:
        grouped['pro_buy_favor_ratio'] = grouped['pro_buy_num'] / grouped['pro_favor_num']
    '''
    #商品热度特征
    timeused = (END_DATE - grouped['time']).map(lambda x: x.days)
    for i in {0, 1, 3, 7, 14, 20}:
        us = grouped[timeused <= i]
        type_cnt = Counter(us['type'])
        grouped['pro_weight_%sday' % i] = 0.015 * type_cnt[1] + 0.6 * type_cnt[2] + 0.3 * type_cnt[4] + \
            1.0 * type_cnt[4] + 0.2*type_cnt[5] + 0.01 * type_cnt[6]
    #交互人数特征
    grouped['pro_user_num'] = grouped['sum'].sum()
    del grouped['time']
    del grouped['type']
    del grouped['sum']
    return grouped


def merge_action_data():
    pro = pd.read_csv(TRAIN_FILE)#读取训练天数的数据
    pro['time'] = pd.to_datetime(pro['time'])
    pro = pro[(pro['time'] >= START_DATE) & (pro['time'] <= END_DATE)]
    pro['sum'] = 1#用于计算交互人数
    grouped = pro[['sku_id', 'type', 'time', 'sum']].groupby('sku_id').apply(add_type_count)
    grouped = grouped.drop_duplicates('sku_id')
    return grouped



#处理评论
def get_from_jdata_comment():
    comment = pd.read_csv('JData_Comment.csv')
    comment['dt'] = pd.to_datetime(comment['dt'])
    comment = comment[comment['dt'] <= END_DATE]
    # find latest comment index
    #选择最近的时间的评价
    idx = comment.groupby(['sku_id'])['dt'].transform(max) == comment['dt']
    comment = comment[idx]
    return comment[['sku_id', 'comment_num', 'has_bad_comment', 'bad_comment_rate']]


def process_product_feat(product_behavior):
    #处理a1到a3
    product_behavior['a1'] = product_behavior['a1'].fillna(0)
    attr1_df = pd.get_dummies(product_behavior['a1'], prefix='a1')
    del product_behavior['a1']
    product_behavior['a2'] = product_behavior['a2'].fillna(0)
    attr2_df = pd.get_dummies(product_behavior['a2'], prefix='a2')
    del product_behavior['a2']
    product_behavior['a3'] = product_behavior['a3'].fillna(0)
    attr3_df = pd.get_dummies(product_behavior['a3'], prefix='a3')
    del product_behavior['a3']
    #处理评价数据
    #chu xian yige wen ti keneng queshao mou xie zhi
    product_behavior['comment_num'] = product_behavior['comment_num'].fillna(-1)
    df = pd.get_dummies(product_behavior['comment_num'], prefix='comment_num')
    del product_behavior['comment_num']
    #是否有差评
    product_behavior['has_bad_comment'] = product_behavior['has_bad_comment'].fillna(-1)
    df2 = pd.get_dummies(product_behavior['has_bad_comment'], prefix='has_bad_comment')
    del product_behavior['has_bad_comment']
    #差评率，直接用nan
    pro_f = pd.concat([attr1_df, attr2_df, attr3_df, df, df2], axis=1)
    product_behavior = pd.concat([product_behavior, pro_f], axis=1)
    return product_behavior
'''
    这里是最关键的地方，可有：
    1、商品最近点击、收藏、加购物车、购买时间(完成)(没用)
    2、商品被点击、收藏、加购物车、购买量(完成)
    3、商品转化率即用户购买量分别除以用户点击、收藏、加购物车这三类行为数（完成）
    5、商品评分（完成）
    6、该商品购买量占该类商品购买量的比值(ic特征，后续提取)
    7、商品近期购买量（3天、7天）（完成）
'''
if __name__ == "__main__":
    product_base = pd.read_csv('JData_Product.csv')#基本的商品特征，包括a1、a2、a3、cate、brand
    product_base = product_base[['sku_id','a1', 'a2', 'a3']]
    comment = get_from_jdata_comment()
    product_behavior = merge_action_data()#提取商品特征
    product_behavior = pd.merge(product_behavior, product_base, on=['sku_id'], how='left')
    product_behavior = pd.merge(product_behavior, comment, on=['sku_id'], how='left')
    #上面两步产生了大量的nan值，下面一步处理nan
    product_behavior = process_product_feat(product_behavior)
    product_behavior.to_csv('./feature/product_feature%s_%s.csv' % (START_DATE, END_DATE), index=None)