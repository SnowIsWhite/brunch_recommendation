import itertools
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
import numpy as np

K=5
def embed(x,emb_dim=K,seed=1, flatten=False, reduce_sum=False):
	if x.dtypes in ['int64','int32','float32','float64']:
		tmp_x=x
		feat_value = tf.constant(x, dtype='float32') 
		feat_value = tf.reshape(feat_value, shape=[-1, 1])
		x=x.apply(str)
	std = 0.001
	minval = -std
	maxval = std

	#feat_value:id matching - like {A: 1, B: 2} here
	t = [doc.split(" ") for doc in x]
	all_values = itertools.chain.from_iterable(t)
	f_dict = {token: idx for idx, token in enumerate(set(all_values))}
 	size = len(f_dict) # the number of unique field value(feature) = p_i
	id_vec=[] 
	for i in range(len(x)):
	    id_vec.append(f_dict[x[i]])


	emb = keras.backend.variable(tf.random.uniform([size, emb_dim], minval, maxval, dtype=tf.float32, seed=seed)) 
	out = tf.nn.embedding_lookup(emb,id_vec) #[None,emb_dim]
    #if flatten:
    #    out = tf.layers.Flatten(out) #[None,emb_dim]
    if reduce_sum:
        out = tf.reduce_sum(out, axis=1) #shape=[None(len(x))] 
    # only for cont var
	try:
	    tmp_x
	except NameError:
		out=out
	else:
    	out = tf.multiply(out, feat_value)
    return out

def embed_keras(x,K, flatten=False, reduce_sum=False):

	emb=keras.layers.Dense(K,weight_init)

    return emb


def _dense_block_mode1( shape, hidden_units, dropouts, densenet=False, training=False, seed=0, bn=False, name="dense_block"):
    """
    :param x:
    :param hidden_units:
    :param dropouts:
    :param densenet: enable densenet
    :return:
    Ref: https://github.com/titu1994/DenseNet
    """
    weight_init = tf.keras.initializers.glorot_uniform()
    inputs = keras.Input(shape=shape)
    z = inputs
    for i, (h, d) in enumerate(zip(hidden_units, dropouts)):
     
        z= keras.layers.Dense(units=h,  kernel_initializer=weight_init)(z)    
        if bn:
            z=keras.layers.BatchNormalization(z)
        z=keras.activations.relu(z)
        #z=keras.activations.selu(z)
        z=keras.layers.Dropout(d,seed=seed * i)(z) if d > 0 else z
        if densenet:
            x = tf.concat([x, z], axis=-1)
        else:
            x = z
    #x = keras.layers.Dense(units=1,  activation=tf.nn.softmax, kernel_initializer=weight_init)(x)    
        
    return x
#    return keras.Model(inputs=inputs, outputs=x)

def dense_block( shape, hidden_units, dropouts, densenet=False, training=False, seed=0, bn=False, name="dense_block"):
    return _dense_block_mode1( shape,hidden_units, dropouts, densenet, training, seed, bn, name)

