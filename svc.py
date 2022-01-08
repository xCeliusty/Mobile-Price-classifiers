#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[2]:


dsT = pd.read_csv('train.csv')
print("Training dataset detials")
dsT.head()


# In[3]:


dsT.info()


# In[4]:


print(dsT['price_range'].value_counts())


# In[5]:


# Get features and target
x = dsT.iloc[:,0:20].values  
y =dsT.iloc[:, 20].values 


# In[6]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.45)


# In[7]:


from sklearn.preprocessing import StandardScaler  
sc = StandardScaler()  
x_train = sc.fit_transform(x_train)  
x_test = sc.transform(x_test) 


# In[8]:


from sklearn.svm import SVC
#svclassifier = SVC(kernel='poly', degree=4)
#svclassifier = SVC(kernel='linear',C=0.1)
svclassifier = SVC(kernel='rbf',C=0.2)
#svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(x_train, y_train)


# In[9]:


y_pred = svclassifier.predict(x_test)


# In[10]:


from sklearn.metrics import classification_report, confusion_matrix
ConfMat=confusion_matrix(y_test, y_pred)
print(ConfMat)


# In[11]:


print(classification_report(y_test, y_pred))


# In[12]:


from sklearn.metrics import accuracy_score

print(' accuracy:')
print((accuracy_score(y_test, y_pred)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




