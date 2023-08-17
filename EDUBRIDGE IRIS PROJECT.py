#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

data = pd.read_csv("IRIS.csv")

print(data)


# In[2]:


print(type(data))


# In[3]:


print(data.head)


# In[4]:


x = data[['sepal_length','sepal_width','petal_length','petal_width']]  #data[:,:-1]
y = data['species']


# In[5]:


X = data.iloc[:,:-1]


# In[6]:


X


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


sp = data['species'].unique()


# In[13]:


sns.FacetGrid(data, hue="species", height=5) \
.map(sns.distplot,"sepal_length")\
.add_legend();

plt.show()


# In[14]:


sns.FacetGrid(data, hue="species", height =5)\
.map(sns.distplot,"petal_length")\
.add_legend();

plt.show()


# In[15]:


sns.FacetGrid(data, hue="species", height =5)\
.map(sns.distplot,"sepal_width")\
.add_legend();

plt.show()


# In[16]:


sns.FacetGrid(data, hue="species", height =5)\
.map(sns.distplot,"petal_width",)\
.add_legend();

plt.show()


# In[17]:


sns.set_style("whitegrid")
sns.FacetGrid(data, hue='species', height =4 ).map(plt.scatter, 'petal_width', 'petal_length')\
.add_legend();

plt.show();


# In[18]:


from sklearn.model_selection import train_test_split                  

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,shuffle=True)


# In[19]:


from sklearn.linear_model import LogisticRegression                


# In[20]:


model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[21]:


from sklearn.metrics import accuracy_score


# In[22]:


lr_acc = accuracy_score(y_test,y_pred)
print(lr_acc)


# In[23]:


from sklearn.neighbors import KNeighborsClassifier


# In[24]:


model = KNeighborsClassifier(n_neighbors=2)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[25]:


KNN_acc = accuracy_score(y_test, y_pred)
KNN_acc


# In[26]:


model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[27]:


KNN_acc = accuracy_score(y_test, y_pred)
KNN_acc


# In[28]:


from sklearn.svm import SVC


# In[29]:


model = SVC()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[30]:


acc_svm = accuracy_score(y_test,y_pred)


# In[31]:


acc_svm


# In[32]:


from sklearn.tree import DecisionTreeClassifier


# In[33]:


model = DecisionTreeClassifier()
model.fit(x_train,y_train)
y_predct = model.predict(x_test)


# In[34]:


acc_dec = accuracy_score(y_test,y_predct)


# In[35]:


acc_dec


# In[ ]:




