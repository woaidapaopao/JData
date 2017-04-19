# -*- coding:utf-8 -*-
#商品特征
import pandas as pd
import numpy as np
from collections import Counter
import datetime as dt
START_DATE = dt.datetime.strptime('2016-2-01',"%Y-%m-%d")#当前时间
END_DATE = dt.datetime.strptime('2016-4-05',"%Y-%m-%d")#当前时间
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

    if type_cnt[1] != 0:
        grouped['pro_latest_browse_time'] = (END_DATE - grouped[grouped['type'] == 1]['time'].max()).days
    else:
        grouped['pro_latest_browse_time'] = np.nan
    if type_cnt[2] != 0:
        grouped['pro_latest_addcart_time'] = (END_DATE - grouped[grouped['type'] == 2]['time'].max()).days
    else:
        grouped['pro_latest_addcart_time'] = np.nan
    if type_cnt[3] != 0:
        grouped['pro_latest_delcart_time'] = (END_DATE - grouped[grouped['type'] == 3]['time'].max()).days
    else:
        grouped['pro_latest_delcart_time'] = np.nan
    if type_cnt[4] != 0:
        grouped['pro_latest_buy_time'] = (END_DATE - grouped[grouped['type'] == 4]['time'].max()).days
    else:
        grouped['pro_latest_buy_time'] = np.nan
    if type_cnt[5] != 0:
        grouped['pro_latest_favor_time'] = (END_DATE - grouped[grouped['type'] == 5]['time'].max()).days
    else:
        grouped['pro_latest_favor_time'] = np.nan
    if type_cnt[6] != 0:
        grouped['pro_latest_click_time'] = (END_DATE - grouped[grouped['type'] == 6]['time'].max()).days
    else:
        grouped['pro_latest_click_time'] = np.nan

    grouped['pro_delcart_addcart_ratio'] = (grouped['pro_delcart_num'] / grouped['pro_addcart_num']) if(grouped['pro_delcart_num'] / grouped['pro_addcart_num'] < 1.) else 1.
    grouped['pro_buy_addcart_ratio'] = (grouped['pro_buy_num'] / grouped['pro_addcart_num']) if (grouped['pro_buy_num'] / grouped['pro_addcart_num'] < 1.).all() else 1.
    grouped['pro_buy_browse_ratio'] = (grouped['pro_buy_num'] / grouped['pro_browse_num']) if (grouped['pro_buy_num'] / grouped['pro_browse_num'] < 1.).all()else 1.
    grouped['pro_buy_click_ratio'] = (grouped['pro_buy_num'] / grouped['pro_click_num']) if (grouped['pro_buy_num'] / grouped['pro_click_num'] < 1.).all() else 1.
    grouped['pro_buy_favor_ratio'] = (grouped['pro_buy_num'] / grouped['pro_favor_num']) if (grouped['pro_buy_num'] / grouped['pro_favor_num'] < 1.).all() else 1.
    #近3、7、10天购买量
    timeused = (END_DATE - grouped['time']).map(lambda x: x.days)
    grouped['buy_3day'] = grouped[(grouped['type'] == 4) & (timeused <= 3)].size
    grouped['buy_7day'] = grouped[(grouped['type'] == 4) & (timeused <= 7)].size
    grouped['buy_10day'] = grouped[(grouped['type'] == 4) & (timeused <= 10)].size

    del grouped['time']
    del grouped['type']
    return grouped

def merge_action_data():#合并用户数据,这里子集和全集都要提取
    pro = pd.read_csv(TRAIN_FILE)#读取训练天数的数据
    pro['time'] = pd.to_datetime(pro['time'])
    pro = pro[(pro['time'] >= START_DATE) & (pro['time'] <= END_DATE)]
    grouped = pro[['sku_id','type','time']].groupby('sku_id').apply(add_type_count)
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
'''
    这里是最关键的地方，可有：
    1、商品最近点击、收藏、加购物车、购买时间(完成)
    2、商品被点击、收藏、加购物车、购买量(完成)
    3、商品转化率即用户购买量分别除以用户点击、收藏、加购物车这三类行为数（完成）
    5、商品评分（完成）
    6、该商品购买量占该类商品购买量的比值
    7、商品近期购买量（3天、7天）（完成）
    8、二次购买的比值
'''
if __name__ == "__main__":
    product_base = pd.read_csv('JData_Product.csv')
    comment = get_from_jdata_comment()
    product_behavior = merge_action_data()#提取商品特征
    product_behavior = pd.merge(product_behavior, product_base, on=['sku_id'], how='left')
    product_behavior = pd.merge(product_behavior, comment, on=['sku_id'], how='left')
    product_behavior.to_csv('./feature/product_feature%s_%s.csv' % (START_DATE, END_DATE), index=None)