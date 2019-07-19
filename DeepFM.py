import nn_module 
from nn_module import embed
import tensorflow as tf
# F : the number of the feature fields
# K : embedding size 


# field : {user, doc, author(categorical_var),views(cont.)} 
# y = "if he read the doc"(0,1) or "the frequency"(0,1,2....)??
# df=pd.DataFrame(data)
""""
df:
user	doc	author	views	y
Jason	q	Jason	4	1	
Molly	M	Jason	24	1
Amy	T	Tina	3100	3
Jake	q	Jason	24 	2
Amy	Ay	Amy	3	0
"""""
data = {'user': ['Jason', 'Molly', 'Amy', 'Jake', 'Amy'], 
        'doc':['q', 'M', 'T', 'q', 'Ay'],
        'author': ['Jason', 'Jason', 'Tina', 'Jason', 'Amy'],
        'views': [4, 24, 3100, 2, 3],
        'y': [1,1,3,1,1]}
df = pd.DataFrame(data)

K=5  #embedding size

# embed 
# e_i := V_i*x_i where  i={1,2,...,},
emb_user = embed(df.user, emb_dim=K, seed=1) # [None , K]
emb_doc = embed(df.doc, emb_dim=K, seed=1)
emb_author = embed(df.author, emb_dim=K, seed=1)
embed_views = embed(df.views, emb_dim=K, seed=1)

emb_list = [
emb_user,
emb_doc,
emb_author,
emb_views
] 

emb_list_d=[]
for emb in emb_list:
    emb_list_d.append(tf.reshape(emb,[-1,1,K]) ) 
embeddings = tf.concat(emb_list_d,axis=1)    #[None, F, K] 
    

# fm layer
fm_list = []

# fm_first order - normal connection (A conneciton with weight to be learned)
## for categorical var including users/docs
bias_user = embed(df.user, emb_dim=1, seed=1) #[None, 1]
bias_doc = embed(df.doc, emb_dim=1, seed=1)
bias_author = embed(df.author, emb_dim=1, seed=1)
bias_views = embed(df.views, emb_dim=1, seed=1)

###여기서 bias 는 first order 에 해당하는 weight값임
fm_first_order_list = [
bias_user,
bias_doc,
bias_author,
bias_views
]
fm_first_order_list = tf.concat(fm_first_order_list, axis=1) #[None, F]
fm_list.append(fm_first_order_list)

# fm_second order - weight-1 connection

emb_concat = embeddings # [None, F , K)]  
emb_sum_squared = tf.square(tf.reduce_sum(emb_concat, axis=1)) #[None,K]
emb_squared_sum = tf.reduce_sum(tf.square(emb_concat), axis=1) #[None,K]

fm_second_order = 0.5 * (emb_sum_squared - emb_squared_sum)
fm_list.extend([emb_sum_squared, emb_squared_sum])

# fm_higher_order interactions
enable_fm_higher_order = False
if enable_fm_higher_order:
	fm_higher_order = dense_block(fm_second_order, hidden_units=[K] * 2,
                                              dropouts=[0.] * 2, densenet=False, seed=1)
	fm_list.append(fm_higher_order)

# Deep component

deep_in = tf.concat(emb_list, axis=1) # [None, (F * K)] 
#hidden_units = [self.params["fc_dim"]*4, self.params["fc_dim"]*2, self.params["fc_dim"]]
hidden_units = [10]*4
dropouts = [0.5] * len(hidden_units)
## 이후 fully connected외 NN도 추가가능 
deep_out = dense_block(shape=deep_in.shape, hidden_units=hidden_units, dropouts=dropouts, densenet=False,
                         seed=1)   #(None, 5, 1)   

#fm_list.append(deep_out)


# DeepFM
fm_first = tf.reduce_sum(fm_first_order_list, axis=1)
fm_second = tf.reduce_sum(fm_second_order, axis=1) #[NONe,]
##맞는지 체크 해야하뮤ㅠ
deep_comp = tf.reduce_sum(tf.reshape(deep_out,[-1,K]), axis=1) #[NONe,]
# this returns x + y.
final = keras.layers.add([fm_first, fm_second, deep_comp]) #[none, ]이어야 

##final output should be [none,]


# model 
###inputs 수정할 것 -- 필요한 모든 input 들어가게 
model = keras.Model(inputs=deep_in, outputs=final)
##Ref:https://stackoverflow.com/questions/46544329/keras-add-external-trainable-variable-to-graph
#/???????
model.layers[0].trainable_weights.extend([
emb_user,
emb_doc,
emb_author,
emb_views
])

model.summary()

model.compile(optimizer=optimizer,
            loss=loss,
            metrics=['accuracy'])
model.fit










