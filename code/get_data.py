#-*- coding:utf-8-*-

import requests
import csv
import pandas as pd
import json
import jieba
import jieba.posseg
import socket
import random
import time

def ment2ent(ment):
    #proxies = {'http' : 'http://' + str(random.choice(ip_pool())[0])}
    #headers = {
    #    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'
    #}
    url_ment = 'http://shuyantech.com/api/cndbpedia/ment2ent?q=' + ment
    res_ent = requests.get(url_ment)
    dict_ent = json.loads(res_ent.text)
    print(dict_ent)
    length = len(dict_ent['ret'])
    if(dict_ent['status'] == 'ok' and length != 0):
        entities = dict_ent['ret']
        return entities
    else:
        print('error')
        return []

def ip_pool():
    IPpool = []
    ips = csv.reader(open('../data/ips.csv'))
    for row in ips:
        IPpool.append(row)
    return IPpool
def avpair(entities, current_id, key_word):
    all_av_dict = {}
    current_av_dict = {}
    #headers = {
    #    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'
    #}
    length = len(entities)
    if(length == 0):
        print('no entity information')
    with open('../data/origin_data2/origin_data' + str(current_id) + '.json', 'a+', encoding='utf-8') as open_file:
        for i in range(length):
            url_avpair = 'http://shuyantech.com/api/cndbpedia/avpair?q=' + entities[i]
            #proxies = {'http' : 'http://' + str(random.choice(ip_pool())[0])}
            res_av = requests.get(url_avpair)
            dict_av = json.loads(res_av.text)
            if dict_av['status'] != 'ok':
                print('information error')
                continue
            else:
                current_data = dict_av['ret']
                current_av_dict[entities[i]] = current_data
            all_av_dict['id'] = current_id
            all_av_dict['key_word'] = key_word
            all_av_dict['content'] = current_av_dict
        json.dump(all_av_dict, open_file, ensure_ascii=False)

def judge_n(passage):
    all_word = set()
    passage_seged = jieba.posseg.cut(passage)
    for word in passage_seged:
        if word.flag == 'n':
            all_word.add(word.word)
    return all_word

def read_txt(path):
    with open(path, 'r') as read_file:
        file_data = read_file.read()
    return file_data

def write_txt(entity, path):
    with open(path, 'a+') as write_file:
        write_file.write(''.join(list(entity)))

def read_entity(path):
    entity_set = set()
    all_df = pd.read_csv(path)
    entity_df = all_df['entityA']
    entity_set = set(list(entity_df))
    return entity_set
if __name__ == '__main__':
    print('begin')
    new_passage = read_txt('../data/seg_list.txt')
    current_id = 1
    old_entity = set()
    old_entity = read_entity('../data/all_data.csv')
    new_entity = set()
    new_entity = judge_n(new_passage)
    for i in new_entity:
        if i in old_entity:
            continue
        try:
            avpair(ment2ent(i), current_id, i)
            current_id += 1
        except:
            print('too many requests')
            time.sleep(3600)
