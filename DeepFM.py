import nn_module 
from nn_module import embed
# F : the number of the feature fields
# F_cat : the number of the categorical feature fields
# K : embedding size 


# field : {user, doc, author(categorical_var),views(cont.)} 
# y = "if he read the doc"(0,1) or "the frequency"(0,1,2....)??
# df=pd.DataFrame(data)
""""
df:
user	doc	author	views	y
Jason	q	Jason	4	1	
Molly	M	Jason	24	1
Amy		T	Tina	31	3
Jake	q	Jason	24 	2
Amy		Ay	Amy		3 	0
"""""
K=5  #embedding size

# embed (only for categorical variables)
# e_i := V_i*x_i where  i={1,2,...,},
emb_user = embed(df.user, emb_dim=K, seed=1) # [None , K]
emb_doc = embed(df.doc, emb_dim=K, seed=1)
emb_author = embed(df.author, emb_dim=K, seed=1)



# fm layer
fm_list = []
# fm_first order - normal connection (A conneciton with weight to be learned)
## for categorical var including users/docs
bias_user = embed(df.user, emb_dim=1, seed=1) #[None, 1]
bias_doc = embed(df.doc, emb_dim=1, seed=1)
bias_author = embed(df.author, emb_dim=1, seed=1)
## for cont. var 
bias_views = tf.constant(df.views, dtype = tf.float32,shape=[len(df.views),1])

fm_first_order_list = [
bias_user,
bias_doc,
bias_author,
bias_views
]
fm_first_order_list = tf.concat(fm_first_order_list, axis=1) #[None, F]


# fm_second order - weight-1 connection
emb_list = [
emb_user,
emb_doc,
emb_author
] 
##for cont var
emb_views = tf.constant(df.views, dtype = tf.float32,shape=[len(df.views),K])
emb_list += emb_views

emb_concat = tf.concat(emb_list, axis=1) # [None, (F_cat * K)]  
emb_sum_squared = tf.square(tf.reduce_sum(emb_concat, axis=1)) #[None,]
emb_squared_sum = tf.reduce_sum(tf.square(emb_concat), axis=1) #[None,]

##cont var 처리 부분 --> ?
fm_second_order = 0.5 * (emb_sum_squared - emb_squared_sum)
fm_list.extend([emb_sum_squared, emb_squared_sum])

# fm_higher_order interactions
enable_fm_higher_order = False
if enable_fm_higher_order:
	fm_higher_order = dense_block(fm_second_order, hidden_units=[K] * 2,
                                              dropouts=[0.] * 2, densenet=False, seed=1)
fm_list.append(fm_higher_order)








