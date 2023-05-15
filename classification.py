#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_excel(r"C:\Users\user\Downloads\iris (4).xls")


# In[3]:


data


# In[5]:


data.isna().sum()


# In[6]:


data.describe()


# In[7]:


data['SL']=data['SL'].fillna(data['SL'].median()) 


# In[8]:


data


# In[9]:


data.isna().sum()


# In[10]:


data['SW']=data['SW'].fillna(data['SW'].median()) 


# In[11]:


data.isna().sum()


# In[12]:


data['PL']=data['PL'].fillna(data['PL'].median()) 


# In[13]:


data


# In[14]:


data.isna().sum()


# In[15]:


for i in['SL','SW','PL','PW']:
    plt.figure()
    plt.boxplot(data[i])
    plt.title(i)


# In[16]:


from sklearn.preprocessing import LabelEncoder


# In[17]:


label_encoder=LabelEncoder()


# In[18]:


data['Classification']=label_encoder.fit_transform(data['Classification'])


# In[19]:


data.head()


# In[20]:


data.tail()


# In[21]:


data


# In[22]:


data['Classification'].unique()


# # logistic regression

# In[56]:


x=data.drop(['Classification'],axis=1)
y=data['Classification']


# In[57]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)


# In[58]:


from sklearn.linear_model import LogisticRegression
logit_model=LogisticRegression()
logit_model.fit(x_train,y_train)
y_pred=logit_model.predict(x_test)


# In[59]:


y_pred


# In[60]:


from sklearn.metrics import confusion_matrix,accuracy_score
print('Accuracy is ',accuracy_score(y_test,y_pred))


# In[61]:


confusion_matrix(y_test,y_pred)


# # knn

# In[29]:


from sklearn.neighbors import KNeighborsClassifier
metric_k=[]
neighbors=np.arange(3,15)


# In[30]:


for k in neighbors:
    classifier=KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2)
    classifier.fit(x_train,y_train)
    y_pred=classifier.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    metric_k.append(acc)


# In[31]:


plt.plot(neighbors,metric_k,'o-')
plt.xlabel('k value')
plt.ylabel('accuracy')
plt.grid()


# In[33]:


classifier=KNeighborsClassifier(n_neighbors=4,metric='minkowski',p=2)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)


# In[34]:


print('Accuracy is',accuracy_score(y_test,y_pred))


# In[35]:


confusion_matrix(y_test,y_pred)


# # svm

# In[62]:


from sklearn.svm import SVC
svmclf=SVC(kernel='linear')
svmclf.fit(x_train,y_train)


# In[63]:


y_pred=svmclf.predict(x_test)


# In[64]:


y_pred


# In[65]:


from sklearn.metrics import accuracy_score,confusion_matrix
print('Accuracy is',accuracy_score(y_test,y_pred))


# In[66]:


from sklearn.svm import SVC
svmclf=SVC(kernel='rbf')
svmclf.fit(x_train,y_train)


# In[67]:


y_pred=svmclf.predict(x_test)


# In[68]:


y_pred


# In[69]:


from sklearn.metrics import accuracy_score,confusion_matrix
print('Accuracy is',accuracy_score(y_test,y_pred))


# In[70]:


confusion_matrix(y_test,y_pred)


# # decision tree

# In[44]:


from sklearn.tree import DecisionTreeClassifier
dt_clf=DecisionTreeClassifier(random_state=42)
dt_clf.fit(x_train,y_train)
y_pred=dt_clf.predict(x_test)


# In[45]:


print('Accuracy is',accuracy_score(y_test,y_pred))


# In[46]:


print(confusion_matrix(y_test,y_pred))


# # random forest

# In[47]:


from sklearn.ensemble import RandomForestClassifier
rf_clf=RandomForestClassifier(random_state=42)
rf_clf.fit(x_train,y_train)


# In[48]:


y_pred=rf_clf.predict(x_test)


# In[49]:


y_pred


# In[50]:


print('Accuracy is',accuracy_score(y_test,y_pred))


# In[51]:


print(confusion_matrix(y_test,y_pred))


# In[52]:


#hyper parametric tuning
rf_c1f1=RandomForestClassifier(n_estimators=20,max_depth=3,criterion='entropy',min_samples_split=2,random_state=42)
rf_c1f1.fit(x_train,y_train)


# In[53]:


y_pred=rf_c1f1.predict(x_test)


# In[54]:


print('Accuracy is',accuracy_score(y_test,y_pred))


# In[55]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:




