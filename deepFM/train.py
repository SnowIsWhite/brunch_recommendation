import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_generation import load_data, get_field_vocab_size
from model import model
##test_df = {idx:{'cola': np.random.randint(1, 100, size=1),
#                 'colb': np.random.randint(1, 100, size=1),
#                 'colc': np.random.randint(1, 100, size=1),
#                'y':np.random.randint(1, 100, size=1)}  for idx in range(100)}
#df = pd.DataFrame(test_df).transpose() #[100,4]
#df.head()

#hyperparamets
hidden_units=[400,400,400,400,400,400]
batch_size = 64
dropout=0.5
embedding_size = 100
lr = 0.001

df = load_data('train', data_num=1000)
print("loading data done")
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]
x_train, x_val, y_train, y_val = train_test_split(X,Y, test_size=0.2, random_state=1)


field_order = list(df.columns)
fvs = get_field_vocab_size()
F=len(df.columns)-1
uv = [fvs[fo]+1 for fo in field_order][:-1]
field_vocab_size = uv
input_data=[]
val_data = []
for i in range(F):
    input_data.append(x_train.iloc[:,i])
    val_data.append(x_val.iloc[:,i])

#train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(
#                buffer_size=len(X_train)).batch(batch_size)
#test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

#hidden 400 400 400/ adam
m = model(embedding_size=embedding_size, field_vocab_size=field_vocab_size, hidden_units=hidden_units, dropout=dropout)
#m.summary()
m.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.Accuracy()])
m.summary()
history = m.fit(input_data,y_train, batch_size=batch_size, epochs=5, validation_data=(val_data, y_val))

m.save('../trained_model.h5')
