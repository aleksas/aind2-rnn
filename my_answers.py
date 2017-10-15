import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation

import keras
import re


# Fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for i in range(len(series) - window_size):#0..4
        X.append(series[i:i + window_size]) #0:4
        y.append(series[i + window_size])

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()

    #layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))
    model.add(LSTM(units=5, input_shape=(window_size,1)))
    #layer 2 uses a fully connected module with one unit
    model.add(Dense(1))

    return model


### return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    lower_case_letters = [chr(i) for i in range(65, 91)]
    upper_case_letters = [chr(i) for i in range(97, 123)]
    valid_chars = punctuation + lower_case_letters + upper_case_letters

    return re.sub(r"[^a-z!,.:;?]", " ", text)

### fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = [text[i:i + window_size] for i in range(0, len(text), step_size) if i + window_size < len(text)]
    outputs = [text[i + window_size] for i in range(0, len(text), step_size) if i + window_size < len(text)]

    return inputs,outputs

# build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss
def build_part2_RNN(window_size, num_chars):
    # 3 layer RNN mode
    model = Sequential()

    #LSTM module with 200 hidden units --> note this should have input_shape = (window_size,len(chars)) where len(chars) = number of unique characters in your cleaned text
    model.add(LSTM(units=200, input_shape=(window_size, num_chars)))
    # a linear module, fully connected, with len(chars) hidden units --> where len(chars) = number of unique characters in your cleaned text
    model.add(Dense(num_chars))
    #softmax activation ( since we are solving a multiclass classification)
    model.add(Activation('softmax'))

    return model
