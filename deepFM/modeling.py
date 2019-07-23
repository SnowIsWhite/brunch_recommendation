import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

##test_df = {idx:{'cola': np.random.randint(1, 100, size=1), 
#                 'colb': np.random.randint(1, 100, size=1), 
#                 'colc': np.random.randint(1, 100, size=1), 
#                'y':np.random.randint(1, 100, size=1)}  for idx in range(100)}
#df = pd.DataFrame(test_df).transpose() #[100,4]
#df.head()


#hyperparamets
batch_size=20
hidden_units=[4,4,4]
dropout=0.5


X=df.iloc[:,:-1]
Y=df.iloc[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=1)



F=len(df.columns)-1
field_vocab_size={i:(df.iloc[:,[i]].max()+1) for i in range(F)}

input_data=[]

for i in range(F):
    input_data.append(X_train.iloc[:,i])

#train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(
#                buffer_size=len(X_train)).batch(batch_size)
#test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))


m = model(embedding_size, field_vocab_size=field_vocab_size, hidden_units=hidden_units, dropout=dropout)
m.summary()
m.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
m.fit(input_data,Y_train, epochs=5)