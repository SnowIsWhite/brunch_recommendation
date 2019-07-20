from nn_module import dense_block, get_embeddings_from_data, fm_block
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

""""
df:
user	doc	author	views	y
Jason	q	Jason	4	    1
Molly	M	Jason	24	    1
Amy	    T	Tina	3100	3
Jake	q	Jason	24 	    2
Amy	    Ay	Amy	    3	    0
"""""
data = {'user': ['Jason', 'Molly', 'Amy', 'Jake', 'Amy'],
        'doc':['q', 'M', 'T', 'q', 'Ay'],
        'author': ['Jason', 'Jason', 'Tina', 'Jason', 'Amy'],
        'views': [4, 24, 3100, 2, 3],
        'y': [1,1,3,1,1]}
df = pd.DataFrame(data)


def create_model(K=5, F=4):
    embeddings= get_embeddings_from_data(df, emb_dim=K) # none, f, k
    dim_one_embeddings = get_embeddings_from_data(df, emb_dim=1)
    dim_one_embeddings = tf.squeeze(dim_one_embeddings)
    deep_in = tf.reshape(embeddings, [-1,F*K]) # [None, (F * K)]
    #hidden_units = [self.params["fc_dim"]*4, self.params["fc_dim"]*2, self.params["fc_dim"]]
    hidden_units = [10]*4
    dropouts = [0.5] * len(hidden_units)
    ## 이후 fully connected외 NN도 추가가능
    fm_out = fm_block(embeddings, dim_one_embeddings) # None, 1
    final_out = dense_block(fm_out, shape=deep_in.shape[1], hidden_units=hidden_units, \
    dropouts=dropouts, densenet=False, seed=1)   #(None, 1)
    return final_out

if __name__ == "__main__":
    res = create_model()
    print(tf.shape(res))
