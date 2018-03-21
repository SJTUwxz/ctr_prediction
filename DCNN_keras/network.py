__author__ = "Xizi Wang"
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate


def buildDCNN(batch_size, vocab_size, word_embeddings_size=48, filter_size=[10,7], nr_of_filters=[6,12], ktop=5, dropout=0.5, output_classes=2):
    sentence = Input(shape=(3,),dtype= 'float32')

