# coding=UTF-8

import pandas as pd
import jieba
import numpy as np
from sklearn.utils import shuffle

model_path = './model/bert_att.h5'
train = pd.read_csv('./data/train.csv')

print(train)
train['sen_cut'] = train['文本'].astype(str).apply(jieba.lcut)

X_train = train['sen_cut'].apply(lambda x: ' '.join(x)).tolist()
y_train = pd.get_dummies((np.asarray(train["情绪标签"])))
text = np.array(X_train)

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

vocab_size = 30000
maxlen = 120

print("开始统计语料的词频信息...")
t = Tokenizer(vocab_size)
t.fit_on_texts(text)
word_index = t.word_index
print('完整的字典大小：', len(word_index))
print("开始序列化句子...")
X_train = t.texts_to_sequences(X_train)
print("开始对齐句子序列...")
X_train = pad_sequences(X_train, maxlen=maxlen, padding='post')
print("完成！")

import copy

small_word_index = copy.deepcopy(word_index)  # 防止原来的字典被改变
x = list(t.word_counts.items())
s = sorted(x, key=lambda p: p[1], reverse=True)
print("移除word_index字典中的低频词...")
for item in s[10000:]:
    small_word_index.pop(item[0])  # 对字典pop
print("完成！")

from bert_serving.client import BertClient

bc = BertClient()
# 定义随机矩阵

embedding_matrix = np.random.uniform(size=(vocab_size + 1, 768))
print("构建embedding_matrix...")
for word, index in small_word_index.items():
    try:
        word_vector = bc.encode([word])
        embedding_matrix[index] = word_vector
        # print("Word: [", index, "]")
    except:
        print("Word: [", word, "] not in wvmodel! Use random embedding instead.")
print("完成！")
print("Embedding matrix shape:\n", embedding_matrix.shape)

from keras.models import Sequential, Model
from keras.layers import Input, Dense, GRU, LSTM, Activation, Dropout, Embedding, Bidirectional
from keras import backend as K
from keras.engine.topology import Layer


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


wv_dim = 768
n_timesteps = maxlen
inputs = Input(shape=(maxlen,))
embedding_sequences = Embedding(vocab_size + 1, wv_dim, input_length=maxlen, weights=[embedding_matrix])(inputs)
lstm = Bidirectional(LSTM(128, return_sequences=True))(embedding_sequences)
l = AttentionLayer()(lstm)
l = Dense(128, activation="tanh")(l)
l = Dropout(0.5)(l)
l = Dense(6, activation="softmax")(l)
m = Model(inputs, l)
m.summary()
m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

m.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.2)
m.save(model_path)
