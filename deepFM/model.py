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
    fm_first_in = backend.squeeze(fm_first_in, axis=2) # none, F

    # dense layer
    dropouts = [dropout] * len(hidden_units)
    weight_init = keras.initializers.glorot_uniform()

    deep_in = layers.Reshape((F*embedding_size,))(embed_list)
    for i, (h, d) in enumerate(zip(hidden_units, dropouts)):
        z = layers.Dense(units=h, kernel_initializer=weight_init)(deep_in)
        z = layers.BatchNormalization(axis=-1)(z)
        z = keras.activations.relu(z)
        z = layers.Dropout(d,seed=d * i)(z) if d > 0 else z
    deep_out = layers.Dense(units=1, activation=tf.nn.softmax, kernel_initializer=weight_init)(z)
    # deep_out: None, 1

    # fm layer
    fm_first_order = backend.sum(fm_first_in, axis=1) #None, 1

    emb_sum_squared = backend.square(backend.sum(embed_list, axis=1)) #none, K
    emb_squared_sum = backend.sum(backend.square(embed_list), axis=1) #none, K
    fm_second_order = layers.Subtract()([emb_sum_squared, emb_squared_sum])
    fm_second_order = backend.sum(fm_second_order, axis=1) #none, 1
    fm_out = layers.Add()([fm_first_order, fm_second_order])

    out = layers.Add()([deep_out, fm_out])
    out = layers.Activation(activation='sigmoid')(out)
    model = keras.Model(inputs=inputs, outputs=out)
    return model
