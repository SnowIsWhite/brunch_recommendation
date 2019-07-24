import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from data_generation import load_data, get_field_vocab_size
from model import model

target_df = './dataframe/test_df_2.csv'
model_name = './trained_model1.h5'

if not os.path.exists(target_df):
    df = load_data('test', data_num=-1)
    print("loading data done")
    df.to_csv(target_df, mode='w')
df= pd.read_csv(target_df, index_col=0)

X=df.iloc[:,:-1]
Y=df.iloc[:,-1]

m = tf.keras.models.load_model(model_name)

result = m.evaluate(X,Y)
