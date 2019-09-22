#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import keras
import time
import jieba
import word2vec
import multiprocessing
import os,yaml
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.recurrent import LSTM,GRU
from keras import Input,Model
from keras.layers.core import Dense, Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.models import model_from_yaml
from keras.utils.vis_utils import plot_model
from keras.layers import concatenate, Reshape, Bidirectional, Permute, multiply
from dailea import train_lstm, LossHistory

start_idx = 70000 #数据集划分标签
end_idx = 100000 #数据集末尾标签

def transword2id(sentence):
    feature_seg = jieba.lcut(str(sentence), cut_all=False)
    word_id = []
    count = 1
    for word in feature_seg:
        if word == ',':
            continue
        if word in word2id_dict:
            word_id.append(word2id_dict[word])
    return word_id

if __name__ == '__main__':
    all_data_df = pd.read_csv('../data/all_data.csv')
    model = word2vec.load('../data/seg_word2vec.bin')
    vocab = model.vocab  
    vocab_num = len(vocab)
    w_num = len(model[vocab[0]])

    embedding_matrix = np.zeros((vocab_num+1,w_num))
    word2id_dict = {}
    for i in range(vocab_num):
        word2id_dict[vocab[i]] = i+1
        embedding_matrix[i+1] = model[vocab[i]]
    print(i)
    word2id_dict[' '] = 0
    embedding_matrix[0] = [0.0 for j in range(100)]

    feature_data_df = all_data_df[['A_attributes', 'A_describe', 'B_attributes', 'B_describe']]
    feature_data_df['A_attributes'] = feature_data_df['A_attributes'].apply(transword2id)
    feature_data_df['B_attributes'] = feature_data_df['B_attributes'].apply(transword2id)
    feature_data_df['A_describe'] = feature_data_df['A_describe'].apply(transword2id)
    feature_data_df['B_describe'] = feature_data_df['B_describe'].apply(transword2id)
    arrA_att = feature_data_df.iloc[:start_idx]['A_attributes'].values
    arrB_att = feature_data_df.iloc[:start_idx]['B_attributes'].values
    arrA_des = feature_data_df.iloc[:start_idx]['A_describe'].values
    arrB_des = feature_data_df.iloc[:start_idx]['B_describe'].values
    arrA_att = sequence.pad_sequences(arrA_att, maxlen=20)
    arrB_att = sequence.pad_sequences(arrB_att, maxlen=20)
    arrA_des = sequence.pad_sequences(arrA_des, maxlen=50)
    arrB_des = sequence.pad_sequences(arrB_des, maxlen=50)
    arrY = all_data_df.iloc[:start_idx]['label'].values

    testA_att = feature_data_df.iloc[start_idx:end_idx]['A_attributes'].values
    testB_att = feature_data_df.iloc[start_idx:end_idx]['B_attributes'].values
    testA_des = feature_data_df.iloc[start_idx:end_idx]['A_describe'].values
    testB_des = feature_data_df.iloc[start_idx:end_idx]['B_describe'].values
    testY = all_data_df.iloc[start_idx:end_idx]['label'].values
    testA_att = sequence.pad_sequences(testA_att, maxlen=20)
    testB_att = sequence.pad_sequences(testB_att, maxlen=20)
    testA_des = sequence.pad_sequences(testA_des, maxlen=50)
    testB_des = sequence.pad_sequences(testB_des, maxlen=50)

    model = train_lstm(vocab_num,w_num,embedding_matrix)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    history = LossHistory()
    model.fit([arrA_att, arrB_att, arrA_des, arrB_des], arrY, batch_size=256, epochs=100,verbose=1,shuffle=True,callbacks=[history])    

    if not os.path.exists('./model'):
        os.makedirs('./model')
    yaml_string = model.to_yaml()
    with open('./model/DAILEA_10000.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('./model/DAILEA_10000.h5')
