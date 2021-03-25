#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# In[2]:


# Importing the dataset
X = pd.read_csv('./Data/X_train_SMOTE.csv')
y = pd.read_csv('./Data/y_train_SMOTE.csv')

# In[3]:


# Splitting the dataset into the Training set and Test set  
# to use sklearn we need data in matrix not in data frame
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[4]:


# Feature Scaling #no need feature scaling for DT as they ar not based on euclidean dist and also will be easy to interpret
from sklearn.preprocessing import StandardScaler #but to visalize with high reolution(0.01),feature scaling helps to execute lot faster
sc = StandardScaler()                       #so we are keeping the feature scaling, but we can also remove feature scaling
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# # Random Forest Classification

# In[8]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion= 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

pickle.dump(classifier,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

# In[9]:

#test_data = pd.read_csv('./test.csv')
# Predicting the Test set results
y_pred = model.predict(X_test)
print(y_pred)

# In[10]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, log_loss, roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
results = confusion_matrix(y_test, y_pred)
print(results)

output = pd.DataFrame(X_test)
y_test.values
output['FraudFound'] = y_pred


output.to_csv(r"./test.csv", index=False)


