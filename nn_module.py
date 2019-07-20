import itertools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
import numpy as np
import pandas as pd

def embed(x, emb_dim=5, seed=1, flatten=False, reduce_sum=False, \
is_categorical=True):
	if x.dtypes in ['int64','int32','float32','float64']:
		feat_value = tf.constant(x, dtype='float32')
		feat_value = tf.reshape(feat_value, shape=[-1, 1])
		x = x.apply(str)
	# param for random values in embedding
	std = 0.001
	minval = -std
	maxval = std

	#feat_value:id matching - like {A: 1, B: 2} here
	t = [doc for doc in x]
	f_dict = {token: idx for idx, token in enumerate(set(t))}
	size = len(f_dict)
	# the number of unique field value(feature) = p_i
	id_vec=[]
	for i in range(len(x)):
		id_vec.append(f_dict[x[i]])

	emb = keras.backend.variable(tf.random.uniform([size, emb_dim], minval, \
	maxval, dtype=tf.float32, seed=seed))
	out = tf.nn.embedding_lookup(emb,id_vec) #[None,emb_dim]
	if not is_categorical:
		out = tf.multiply(out, feat_value)
	return out

def get_embeddings_from_data(df, emb_dim=5):
	K=emb_dim  #embedding sizw
    # embed
    # e_i := V_i*x_i where  i={1,2,...,},
	emb_user = embed(df.user, emb_dim=K, seed=1)# [None , K]
	emb_doc = embed(df.doc, emb_dim=K, seed=1)
	emb_author = embed(df.author, emb_dim=K, seed=1)
	emb_views = embed(df.views, emb_dim=K, seed=1, is_categorical=False)

	emb_list = [
    emb_user,
    emb_doc,
    emb_author,
    emb_views
    ]
	emb_list_d=[]
	for emb in emb_list:
		emb_list_d.append(tf.reshape(emb,[-1,1,K]) ) # None, 1, K
	embeddings = tf.concat(emb_list_d,axis=1)    #[None, F, K]
	return embeddings

def _dense_block_mode1(fm_out, shape, hidden_units, dropouts, densenet=False, \
training=False, seed=0, bn=False, name="dense_block"):
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
		z = keras.layers.Dense(units=h, kernel_initializer=weight_init)(z)
		if bn:
			z=keras.layers.BatchNormalization(z)
		z = keras.activations.relu(z)
		#z=keras.activations.selu(z)
		z=keras.layers.Dropout(d,seed=seed * i)(z) if d > 0 else z
		if densenet:
			x = tf.concat([x, z], axis=-1)
		else:
			x = z
	x = keras.layers.Dense(units=1, activation=tf.nn.softmax, \
	kernel_initializer=weight_init)(x)
	x = keras.layers.add([x, fm_out])
	x = keras.activations.sigmoid(x)
	print(x.shape)
	return x

def dense_block(fm_out, shape, hidden_units, dropouts, densenet=False, \
training=False, seed=0, bn=False, name="dense_block"):
    return _dense_block_mode1(fm_out, shape,hidden_units, dropouts, \
	densenet, training, seed, bn, name)

def fm_block(embeddings, dim_one_embeddings):
	# first order
	fm_first_order = tf.reduce_sum(dim_one_embeddings,axis=1) # None, F -> None,1

	# second order
	emb_sum_squared = tf.square(tf.reduce_sum(embeddings, axis=1)) #none, k
	emb_squared_sum = tf.reduce_sum(tf.square(embeddings), axis=1)
	fm_second_order = 0.5 * (emb_sum_squared - emb_squared_sum)
	fm_second_order = tf.reduce_sum(fm_second_order, axis=1) #none, k => none, 1
	return tf.add(fm_first_order, fm_second_order) # none, 1
