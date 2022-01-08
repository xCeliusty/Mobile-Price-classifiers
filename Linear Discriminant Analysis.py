#!/usr/bin/env python
# coding: utf-8

# In[72]:


# necessary import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
  


# In[73]:


dataset = pd.read_csv('train.csv')
dataset.info()


# In[74]:


x = dataset.iloc[:,0:20].values  
y = dataset.iloc[:, 20].values 


# In[111]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.45, random_state =42)  


# In[112]:


from sklearn.preprocessing import StandardScaler  
sc = StandardScaler()  
x_train = sc.fit_transform(x_train)  
x_test = sc.transform(x_test) 


# In[113]:


lda = LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)
y_pred = lda.predict(x_test) 


# In[114]:


#print the accuracy and confusion matrix
print('Accuracy : ' + str(accuracy_score(y_test, y_pred)))
conf_m = confusion_matrix(y_test, y_pred)
print(conf_m)


# In[115]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[116]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

