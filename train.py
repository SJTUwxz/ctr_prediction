# -*- coding: utf-8 -*-
import keras
import numpy as np
import pandas as pd
import os
import gensim
import jieba
import re
import time
import random
from numpy import array
from keras.preprocessing import text, sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Model 
from keras.layers import Dense, Input, Embedding, Conv2D, MaxPool2D
from keras.layers import Flatten, Reshape, Concatenate, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback, ModelCheckpoint, ProgbarLogger
from gensim.models import Word2Vec
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

f = open('/data/users/xiziwang/ctr_data/stopwords.txt.utf8','r')
stopwords = [l.strip() for l in f.readlines()]

def word_div(string):
    string = string.decode( 'utf-8')
    filtrate = re.compile(u'[^\u4E00-\u9FA5]')
    reContent = filtrate.sub(r'', string)
    string = string.encode( 'utf-8')
    string = re.sub('\d', '', string)
    string = re.sub('\s', '', string) 
    string_seg = jieba.cut(string)
    new_string = ""
    for word in string_seg:
        word = word.encode( 'utf-8') 
        if word not in stopwords:
            new_string += " "
            new_string += word
    return new_string


def process_csv():
    f = '/data/users/xiziwang/ctr_data/news_quality_dataset_temp_result' 
    if os.path.exists(f) == False:
        return False
    table = pd.read_csv(f, delimiter='\x01')
    titles = []
    ctr_rate = []
    for i,row in table.iterrows() :
        try:
            click = int(row[3]) 
            show = int(row[4]) 
            num = float(float(click)/ float(show))
        except:
            continue
        title = word_div(row[1] ) 
        titles.append(title)
        if(num< 0.1) :
            ctr_rate.append(0)
        else:
            ctr_rate.append(1) 
    return titles, ctr_rate

def get_model(embedding_matrix, maxlen=10, max_features=100000, embed_size=48,filter_sizes = [1,2,3,5], num_filters=32 ):
    inp = Input(shape=(maxlen,) ) 
    x = Embedding(max_features, embed_size, weights = [embedding_matrix])(inp)  
    x = SpatialDropout1D(0.4)(x)  
    x = Reshape((maxlen, embed_size, 1) )(x)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size ), kernel_initializer= 'normal', activation= 'relu' )(x) 

    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size ), kernel_initializer= 'normal', activation= 'relu' )(x) 

    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size ), kernel_initializer= 'normal', activation= 'relu' )(x) 

    conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size ), kernel_initializer= 'normal', activation= 'relu' )(x) 

    maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] +1 , 1) )(conv_0)  

    maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] +1 , 1) )(conv_1)  

    maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] +1 , 1) )(conv_2)  

    maxpool_3 = MaxPool2D(pool_size=(maxlen - filter_sizes[3] +1 , 1) )(conv_3)  

    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3] ) 
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(1, activation= 'sigmoid')(z)

    model = Model(inputs = inp, outputs = outp)
    model.compile(loss= 'binary_crossentropy', optimizer= 'adam', metrics=[ 'accuracy'] ) 

    return model

def RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval = 1 ):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score) ) 


if __name__ == '__main__':
    
    t = time.time() 
    titles, ctr = process_csv()
    process_time = time.time() - t 

    print ( 'Preprocess time: {}'.format(process_time) ) 
    length = len(titles)
    
    # shuffle two lists
    combined = list(zip(titles, ctr) )
    random.shuffle(combined)
    titles[:], ctr[:] = zip(*combined)   

    train_titles = titles[:int(length * 0.8) ] 
    val_titles = titles[int(length * 0.8): int(length * 0.9)]
    test_titles = titles[int(length * 0.9): ] 

    train_ctr = ctr[:int(length * 0.8) ] 
    val_ctr = ctr[int(length * 0.8): int(length * 0.9)]
    test_ctr = ctr[int(length * 0.9): ] 

    
    t_time = time.time() 
    t = Tokenizer() 
    t.fit_on_texts(train_titles)
    vocab_size = len(t.word_index) + 1 
    encoded_docs = t.texts_to_sequences(train_titles)

    token_time = time.time() - t_time 
    print ( 'Token time: {}'.format(token_time) ) 
    max_length = 10
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length,padding= 'post')

    embeddings_index = dict()
    t_time = time.time() 
    print 'loading word2vec model'
    
    w2vmodel = Word2Vec.load( '/data/users/xiziwang/ctr_data/word2vec.model' )
    
    for word in w2vmodel.wv.vocab:
        try:
            embeddings_index[word] = w2vmodel[word]  
        except:
            continue
    # print( 'Loaded {} word vectors.'.format(embeddings_index))
    print ('Load word2vec time: {}'.format(time.time() - t_time ) ) 
    embedding_matrix = np.zeros((vocab_size, 48 ))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    model = get_model(embedding_matrix = embedding_matrix, max_features = vocab_size)

    batch_size = 256
    epochs = 20 
    checkpoint = ModelCheckpoint('/data/users/xiziwang/ctr_data/models/ctr-{epoch:02d}-{val_loss:05f}.h5', monitor= 'val_acc', verbose=1, save_best_only =False)
    callback_list = [checkpoint]  
    model.fit(padded_docs, train_ctr, epochs=epochs, verbose=1, callbacks = callback_list)
    loss,accuracy = model.evaluate(padded_docs, train_ctr, verbose = 0)
    print( 'Accuracy: %f' % (accuracy * 100) ) 
    
