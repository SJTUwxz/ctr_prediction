import time
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import random

def gen_dataset(filename):
    f = open(filename, 'r')
    dictionary = {}
    titles = [ 'title', 'content', 'click', 'show' ]
    newses = f.read().splitlines()
    length = len(newses) 
    t0 = time.time() 
    for i,news in enumerate(newses):
        contents = news.split( '\x01') 
        if len(contents) != 6:
            continue
        dictionary[contents[0]] = { 
                'title':contents[1].decode( 'utf-8'),
                'content':contents[2].decode( 'utf-8'),
                'click': int(contents[3]),
                'show':int(contents[4]) 
        }   
    t1 = time.time()
    print t1-t0
   
    keys = list(dictionary.keys()) 
    random.shuffle(keys) 
    train_num = int(length * 0.8)
    val_num = int(length * 0.1)
    train_keys = keys[:train_num]
    val_keys = keys[train_num:train_num+val_num]
    test_keys = keys[train_num+val_num:]

    # train_csv.to_csv( '/home/xiziwang/projects/ctr_pred/data/news_quality_dataset_train.csv')
    # val_csv.to_csv( '/home/xiziwang/projects/ctr_pred/data/news_quality_dataset_val.csv')
    # test_csv.to_csv( '/home/xiziwang/projects/ctr_pred/data/news_quality_dataset_test.csv')

if __name__ == '__main__':
    gen_dataset('../../data/news_quality_dataset_temp_result') 
            


