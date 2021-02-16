#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
train=pd.read_csv('Salary_Train.csv')
train.head()


# In[2]:


train.tail()


# In[3]:


train.shape


# In[4]:


train.isnull().any()


# In[5]:


train.isnull().any().sum()


# In[7]:


train.duplicated().any()


# In[8]:


train[train.duplicated()]


# In[10]:


train.drop_duplicates(inplace=True)


# In[11]:


train.shape


# In[13]:


train.duplicated().sum()


# In[14]:


train.info()


# In[15]:


train['Salary'].unique()


# In[16]:


train['Salary'].value_counts().plot(kind='bar')


# In[17]:


test=pd.read_csv('Salary_Test.csv')
test.head()


# In[18]:


test.tail()


# In[19]:


test.isnull().any()


# In[20]:


test.isnull().any().sum()


# In[21]:


test.shape


# In[22]:


test.duplicated().any()


# In[23]:


test[test.duplicated()]


# In[25]:


test.drop_duplicates(inplace=True)


# In[26]:


test.duplicated().sum()


# In[27]:


test.shape


# In[28]:


test['Salary'].unique()


# In[29]:


test.info()


# In[30]:


test['Salary'].value_counts().plot(kind='bar')


# In[31]:


string_col=['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()


# In[32]:


for i in string_col:
    train[i]=label.fit_transform(train[i])
    test[i]=label.fit_transform(test[i])


# In[33]:


train


# In[34]:


test


# In[ ]:





# In[46]:


from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
train=scale.fit_transform(train)
test=scale.fit_transform(test)


# In[47]:


train=pd.DataFrame(train)


# In[48]:


train


# In[49]:


test=pd.DataFrame(test)


# In[50]:


test


# # Linear SVM

# In[51]:


trainx=train.iloc[:,0:13]
trainy=train.iloc[:,13]
testx=test.iloc[:,0:13]
testy=test.iloc[:,13]


# In[52]:


from sklearn.svm import SVC
lin_model=SVC(kernel='linear')
lin_model.fit(trainx,trainy)


# In[53]:


lin_model.predict(testx)


# In[54]:


testy


# In[55]:


#Test accuracy
import numpy as np
np.mean(lin_model.predict(testx)==testy)


# In[56]:


#Train accuracy
np.mean(lin_model.predict(trainx)==trainy)


# In[ ]:





# # Poly SVM

# In[57]:


poly_model=SVC(kernel='poly')
poly_model.fit(trainx,trainy)


# In[58]:


poly_model.predict(testx)


# In[59]:


testy


# In[60]:


#Test accuracy
np.mean(poly_model.predict(testx)==testy)


# In[61]:


#Train accuracy
np.mean(poly_model.predict(trainx)==trainy)


# In[ ]:





# # rbf SVM

# In[62]:


rbf_model=SVC(kernel='rbf')
rbf_model.fit(trainx,trainy)


# In[63]:


rbf_model.predict(testx)


# In[64]:


testy


# In[65]:


#Test accuracy
np.mean(rbf_model.predict(testx)==testy)


# In[66]:


#Train accuracy
np.mean(rbf_model.predict(trainx)==trainy)


# In[ ]:





# In[67]:


#visualization
import matplotlib.pyplot as plt
plt.xlabel('Train_y')
plt.ylabel('Test_y')
plt.plot(trainy)
plt.plot(testy)
plt.legend(['Train_res','Test_res'],loc='lower right')
plt.show()


# In[ ]:




