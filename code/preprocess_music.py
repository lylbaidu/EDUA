import os
import pickle
import argparse

from itertools import islice
import numpy as np
import pandas as pd
import random
import json
from collections import defaultdict
from sklearn import preprocessing

def load_interaction(data_dir):
    all_data = []
    f = open(data_dir, 'r')
    user_src2tgt_dict, item_src2tgt_dict = {}, {}
    for eachline in f:
        eachline = json.loads(eachline)
        u, i, r = eachline['reviewerID'], eachline['asin'], eachline['overall']
        if u not in user_src2tgt_dict:
            user_src2tgt_dict[u] = len(user_src2tgt_dict) + 1  # 从1编码
        user = user_src2tgt_dict[u]
        if i not in item_src2tgt_dict:
            item_src2tgt_dict[i] = len(item_src2tgt_dict) + 1
        item = item_src2tgt_dict[i]
        all_data.append([user, item, r])
    return all_data, user_src2tgt_dict, item_src2tgt_dict

def load_meta(f_meta, item_src2tgt_dict):
    cate_src2tgt_dict = {}
    cate_src_count = defaultdict(int)
    item_cate_dict = {}
    item_cate_dict[0] = 0
    f = open(f_meta, 'r', encoding='utf-8')
    for eachline in f:
        eachline = json.dumps(eval(eachline))
        eachline = json.loads(eachline)

        i, c = eachline['asin'], eachline['categories']
        if i in item_src2tgt_dict:
            for tmp_c in c:
                if tmp_c[0] == 'Digital Music':
                    break
            category = tmp_c[-1]
            if category not in cate_src2tgt_dict:
                cate_src2tgt_dict[category] = len(cate_src2tgt_dict) + 1
            cate = cate_src2tgt_dict[category]
            cate_src_count[category] += 1
            item_cate_dict[item_src2tgt_dict[i]] = cate   #item_id -> cate_id

    item_cate_dict = {k: v for k, v in sorted(item_cate_dict.items(), key=lambda kv: (kv[0], kv[1]))}
    return cate_src2tgt_dict, item_cate_dict

def create_user_list(all_data, user_size):
    user_list = [dict() for u in range(user_size + 1)]
    count = 0
    for u,i,w in all_data:
        user_list[u][i] = w
        count += 1
    print('interaction num:', count)
    return user_list

def split_train_test(user_list, test_size=0.2, time_order=False):
    train_user_list = [None] * len(user_list)
    test_user_list = [None] * len(user_list)
    for user, item_dict in enumerate(user_list):
        if time_order:
            # Choose latest item
            item = sorted(item_dict.items(), key=lambda x: x[1], reverse=True)
            latest_item = item[:int(len(item)*test_size)]
            assert max(item_dict.values()) == latest_item[0][1]
            test_item = set(map(lambda x: x[0], latest_item))
        else:
            # Random select
            test_item = set(np.random.choice(list(item_dict.keys()),
                                             size=int(len(item_dict)*test_size),
                                             replace=False))
        #if user>0:
        #    assert (len(test_item) > 0), "No test item for user %d" % user
        if user>0 and len(test_item)==0:
            print("No test item for user %d" % user)
        test_user_list[user] = test_item
        train_user_list[user] = set(item_dict.keys()) - test_item
    return train_user_list, test_user_list

def create_pair(user_list):
    pair = []
    for user, item_set in enumerate(user_list):
        pair.extend([(user, item) for item in item_set])
    return pair

def instance_a_eval_loader(user_num, item_num, train_user_list, test_user_list, test_neg):  #可以划分训练测试的时候先把test neg sampling 先固定掉，这样每次和baseline比较也更公平
    users, items, ratings = [], [], []
    for u, ilist in enumerate(test_user_list):
        if len(ilist) == 0:
            continue
        neg_num = test_neg * len(ilist)
        neg_set_all = set(range(item_num)) - ilist - train_user_list[u] #uesr train/test都没有买过的item list
        if neg_num >= len(neg_set_all):
            neg_list = neg_set_all
        else:
            neg_list = np.random.choice(list(neg_set_all), neg_num, replace=False)

        #neg_list = np.random.choice(list(set(range(item_num)) - ilist - train_user_list[u]), test_neg-len(ilist))
        #test_list = np.append(ilist, neg_list)
        #test_list = np.random.shuffle(test_list)
        u_test_list = [(i, 0) for i in neg_list] + [(i,1) for i in ilist]
        u_test_list = sorted(u_test_list, key=lambda x: x[0])
        for i,r in u_test_list:
            users.append(u)
            items.append(i)
            ratings.append(r)
    return list(zip(users, items, ratings))

if __name__ == '__main__':
    random.seed(2020)
    np.random.seed(2020)
    f_interation = '../music/Digital_Music_5.json'
    f_meta = '../music/meta_Digital_Music.json'
    f_out = '../music/music_data.pkl'
    test_ratio = 0.2

    all_data, user_src2tgt_dict, item_src2tgt_dict = load_interaction(f_interation)  # 加载数据，得到原id->正式id的dict
    user_size = len(user_src2tgt_dict)
    item_size = len(item_src2tgt_dict)

    cate_src2tgt_dict, item_cate_dict = load_meta(f_meta, item_src2tgt_dict)
    cate_size = len(cate_src2tgt_dict)
    print('user_num, item_num, cate_num', len(user_src2tgt_dict), len(item_src2tgt_dict), len(cate_src2tgt_dict))  # 5541, 3568, 60

    total_user_list = create_user_list(all_data, user_size) #把[u,i,r]转为list[u]={item...}
    print('avg. items per user: ', np.mean([len(u) for u in total_user_list]))  #11.68


    # ----------------------------------partition
    train_user_list, test_user_list = split_train_test(total_user_list,  # 划分train/test
                                                           test_size=test_ratio)
    print('min len(user_itemlist): ', min([len(ilist) for ilist in train_user_list[1:]]))

    train_pair = create_pair(train_user_list)
    print('Complete creating pair')

    dataset = {'user_size': user_size+1, 'item_size': item_size+1,
               'train_user_list': train_user_list, 'test_user_list': test_user_list,
               'cate_size': cate_size+1, 'item_cate_dict': item_cate_dict,
               'train_pair': train_pair}
    with open(f_out, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)




