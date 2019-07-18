import itertools

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
import numpy as np

#emb_dim=K

def embed(x,emb_dim=K,seed=1, flatten=False, reduce_sum=False):
	std = 0.001
	minval = -std
	maxval = std
	#feat_value:id matching - like {A: 1, B: 2} here
	t = [doc.split(" ") for doc in x]
	all_values = itertools.chain.from_iterable(t)
	f_dict = {token: idx if not token.isdigit() else int(token)
	             for idx, token in enumerate(set(all_values))}
 	size = len(f_dict) # the number of unique field value(feature) = p_i
	id_vec=[] 
	for i in range(len(x)):
	    id_vec.append(f_dict[x[i]])

	emb = tf.Variable(tf.random.uniform([size, emb_dim], minval, maxval, dtype=tf.float32, seed=seed)) 
	out = tf.nn.embedding_lookup(emb,id_vec) #[None,emb_dim]
    #if flatten:
    #    out = tf.layers.Flatten(out) #[None,emb_dim]
    if reduce_sum:
        out = tf.reduce_sum(out, axis=1) #shape=[None(len(x))] 
    return out


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
        z=keras.layers.Dropout(d,seed=seed * i)(z,training=training) if d > 0 else z
        if densenet:
            x = tf.concat([x, z], axis=-1)
        else:s
            x = z
    return keras.Model(inputs=inputs, outputs=x)

def dense_block(x, hidden_units, dropouts, densenet=False, training=False, seed=0, bn=False, name="dense_block"):
    return _dense_block_mode1(x, hidden_units, dropouts, densenet, training, seed, bn, name)


model = create_model()
model.summary()


final_model=[]
final_model.append() ###first+second+
model.compile(optimizer=optimizer,
            loss=loss,
            metrics=['accuracy'])






