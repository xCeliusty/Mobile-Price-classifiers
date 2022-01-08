#!/usr/bin/env python
# coding: utf-8

# In[10]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[11]:


print(tf.__version__)
print(keras.__version__)


# In[12]:


dataset = pd.read_csv('mobilePriceClassification.csv')
print(dataset.columns)
dataset.head(10)


# In[13]:


X = dataset.iloc[:,:20].values
y = dataset.iloc[:,20:21].values
print(X[0])
print(y[:10])
sc = StandardScaler()
X = sc.fit_transform(X)
print('Normalized data:')
print(X[0])
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()
print('One hot encoded array:')
print(y[:10])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)


# In[16]:


model = keras.Sequential()
model.add(keras.layers.Dense(16, input_dim=20, activation='relu'))
model.add(keras.layers.Dense(12, activation='relu'))
model.add(keras.layers.Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, batch_size=80)
model.summary()


# In[17]:


predications = model.predict(X_test)
y_pred = list()
for i in range(len(predications)):
    y_pred.append(np.argmax(predications[i]))

ytest = list()
for i in range(len(y_test)):
    ytest.append(np.argmax(y_test[i]))

print('Accuracy is:', accuracy_score(ytest, y_pred)*100)


# In[ ]:





# In[ ]:




