import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers
import pandas as pd

def model(embedding_size, field_vocab_size=[], hidden_units=[4,4,4], dropout=0.5):
    F = len(field_vocab_size)

    # prepare embeddings
    inputs = []
    embed_list = []
    embed_one_list = []
    for i, vocab_size in enumerate(field_vocab_size):
        in_ = keras.Input(shape=(1,))
        inputs.append(in_)
        embed_list.append(layers.Embedding(vocab_size, embedding_size, input_length=1)(in_))
        embed_one_list.append(layers.Embedding(vocab_size, 1, input_length=1)(in_))
    embed_list = layers.concatenate(embed_list, axis=1) # none, F, K

    fm_first_in = layers.concatenate(embed_one_list, axis=1) # None, F, 1
    fm_first_in = layers.Lambda(lambda x: backend.squeeze(x, axis=2))(fm_first_in)

    # dense layer
    dropouts = [dropout] * len(hidden_units)
    weight_init = keras.initializers.glorot_uniform()

    deep_in = layers.Reshape((F*embedding_size,))(embed_list)
    for i, (h, d) in enumerate(zip(hidden_units, dropouts)):
        z = layers.Dense(units=h, kernel_initializer=weight_init)(deep_in)
        z = layers.BatchNormalization(axis=-1)(z)
        z = layers.Activation("relu")(z)
        z = layers.Dropout(d,seed=d * i)(z) if d > 0 else z
    deep_out = layers.Dense(units=1, activation=tf.nn.softmax, kernel_initializer=weight_init)(z)
    # deep_out: None, 1

    # fm layer
    fm_first_order = layers.Lambda(lambda x: backend.sum(x, axis=1))(fm_first_in) #None, 1

    emb_sum_squared = layers.Lambda(lambda x: backend.square(backend.sum(x, axis=1)))(embed_list) #none, K
    emb_squared_sum = layers.Lambda(lambda x: backend.sum(backend.square(x), axis=1))(embed_list) #none, K
    fm_second_order = layers.Subtract()([emb_sum_squared, emb_squared_sum])
    fm_second_order = layers.Lambda(lambda x: backend.sum(x, axis=1))(fm_second_order) #none, 1
    fm_out = layers.Add()([fm_first_order, fm_second_order])

    out = layers.Add()([deep_out, fm_out])
    out = layers.Activation(activation='sigmoid')(out)
    model = keras.Model(inputs=inputs, outputs=out)
    return model
