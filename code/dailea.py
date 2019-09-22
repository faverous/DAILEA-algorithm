#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import keras
import time
import jieba
import word2vec
import multiprocessing
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

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
 
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def attention_word_block(inputs):
    #input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(20, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_word')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul_word')
    return output_attention_mul

def attention_sen_block(inputs):
    #input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(50, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_sen')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul_sen')
    return output_attention_mul

def train_lstm(vocab_num, w_num, embedding_matrix):
    print ('Defining a Simple Keras Model...')
    
    ## 输入层
    inputA_att = Input(shape=(20,), dtype='int32', name='inputA_att')
    inputB_att = Input(shape=(20,), dtype='int32', name='inputB_att')
    inputA_des = Input(shape=(50,), dtype='int32', name='inputA_des')
    inputB_des = Input(shape=(50,), dtype='int32', name='inputB_des')
    
    ## 编码层
    embedA_att = Embedding(input_dim=vocab_num+1, output_dim=w_num,weights = [embedding_matrix], trainable = False, name='embedA_att')(inputA_att)
    embedB_att = Embedding(input_dim=vocab_num+1, output_dim=w_num,weights = [embedding_matrix], trainable = False, name='embedB_att')(inputB_att)
    embedA_des = Embedding(input_dim=vocab_num+1, output_dim=w_num,weights = [embedding_matrix], trainable = False, name='embedA_des')(inputA_des)
    embedB_des = Embedding(input_dim=vocab_num+1, output_dim=w_num,weights = [embedding_matrix], trainable = False, name='embedB_des')(inputB_des)
    
    ## 词类别
    lstmA_att = Bidirectional(LSTM(units=32, activation='tanh', name='lstm_a_att', return_sequences=True), name='bi_a_att')(embedA_att)
    lstmB_att = Bidirectional(LSTM(units=32, activation='tanh', name='lstm_b_att', return_sequences=True), name='bi_b_att')(embedB_att)
    embedAB_att = concatenate([lstmA_att, lstmB_att])
    attention_mul = attention_word_block(embedAB_att)
    attention_flatten = Flatten()(attention_mul)
    #AB_att = Reshape(target_shape=(64,1))(attention_flatten)
    #lstmAB_att = LSTM(units=64, activation='tanh', name='lstm_ab_att')(attention_flatten)
    
    ## 句类别
    lstmA_des = Bidirectional(LSTM(units=32, activation='tanh', name='lstm_a_des', return_sequences=True), name='bi_a_des')(embedA_des)
    lstmB_des = Bidirectional(LSTM(units=32, activation='tanh', name='lstm_b_des', return_sequences=True), name='bi_b_des')(embedB_des)
    embedAB_des = concatenate([lstmA_des, lstmB_des])
    attention_mul_des = attention_sen_block(embedAB_des)
    attention_flatten_des = Flatten()(attention_mul_des)
    
    ## 汇总
    A_B = concatenate([attention_flatten, attention_flatten_des])
    A_B = Dense(units=32, activation='tanh', name='lstm_c')(A_B)
    den = Dense(2)(A_B) # Dense=>全连接层,输出维度=2
    ans = Activation(activation='softmax', name='softmax')(den)
    print(ans.shape)
    model = Model(inputs=[inputA_att, inputB_att, inputA_des, inputB_des], outputs=[ans])
    model.summary()
    return model