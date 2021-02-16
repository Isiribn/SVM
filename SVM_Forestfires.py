#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv('forestfires_svm.csv')
data.head()


# In[2]:


data.shape


# In[3]:


data.tail()


# In[4]:


data.isnull().any()


# In[5]:


data.isnull().any().sum()


# In[6]:


data.drop('month',inplace=True,axis=1)


# In[7]:


data.drop('day',inplace=True,axis=1)


# In[8]:


data.head()


# In[9]:


data.shape


# In[10]:


data['size_category']


# In[11]:


data.size_category.unique()


# In[12]:


data.loc[data.size_category=='small','size_category']=0


# In[13]:


data.loc[data.size_category=='large','size_category']=1


# In[14]:


data['size_category'].unique()


# In[15]:


data.info()


# In[16]:


data.describe()


# In[18]:


data.shape


# In[31]:


data['size_category']=data['size_category'].astype('int')


# In[32]:


data.info()


# In[33]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=0.3)


# In[34]:


trainx=train.iloc[:,0:28]
trainy=train.iloc[:,28]
testx=test.iloc[:,0:28]
testy=test.iloc[:,28]


# In[21]:


help(SVC)


# # Linear kernel

# In[35]:


lin_model=SVC(kernel='linear')


# In[36]:


lin_model.fit(trainx,trainy)


# In[37]:


pred_lin=lin_model.predict(testx)


# In[38]:


pred_lin


# In[39]:


testy


# In[41]:


#testing accuracy for linear SVM
import numpy as np
np.mean(pred_lin==testy)


# In[42]:


#training accuracy for linear SVM
np.mean(lin_model.predict(trainx)==trainy)


# # poly kernel

# In[43]:


poly_model=SVC(kernel='poly')
poly_model.fit(trainx,trainy)


# In[44]:


poly_model.predict(testx)


# In[45]:


testy


# In[46]:


#Train accuracy for poly 
np.mean(poly_model.predict(trainx)==trainy)


# In[47]:


#Test accuracy for poly
np.mean(poly_model.predict(testx)==testy)


# # rbf kernel

# In[48]:


rbf_model=SVC(kernel='rbf')
rbf_model.fit(trainx,trainy)


# In[49]:


rbf_model.predict(testx)


# In[50]:


testy


# In[51]:


#Train accuracy for rbf
np.mean(rbf_model.predict(trainx)==trainy)


# In[52]:


#test accuracy for rbf
np.mean(rbf_model.predict(testx)==testy)


# In[55]:


import seaborn as sns
sns.hist(data['size_category'],kind='bar')


# In[57]:


import matplotlib.pyplot as plt
plt.plot(poly_model.predict(testx))
plt.plot(testy)
plt.title('Predicted vs Actual')
plt.legend(['Pred','Actual'],loc='lower right')
plt.show()


# In[60]:


import matplotlib.pyplot as plt
plt.plot(rbf_model.predict(testx))
plt.plot(testy)
plt.title('Predicted vs Actual')
plt.legend(['Pred','Actual'],loc='lower right')
plt.show()


# In[61]:


import matplotlib.pyplot as plt
plt.plot(lin_model.predict(testx))
plt.plot(testy)
plt.title('Predicted vs Actual')
plt.legend(['Pred','Actual'],loc='lower right')
plt.show()


# In[ ]:




