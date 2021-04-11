#!/usr/bin/env python
# coding: utf-8

# # Importing dataset

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# # Importing dataset

# In[2]:


dataset=pd.read_csv(r'G:\Study Material\Projects\Iris Flower Dataset\Classification\Iris.csv')
dataset


# In[3]:


dataset.info()


# In[4]:


dataset.isnull().sum()


# In[5]:


copy=pd.DataFrame(dataset)


# In[6]:


copy=copy.drop(['Id'],axis=1)


# In[7]:


copy


# In[8]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
copy['Species']=le.fit_transform(copy['Species'])


# In[9]:


copy


# In[10]:


df=pd.DataFrame(copy)


# In[11]:


df


# In[12]:


x=df.iloc[:,0:4].values


# In[13]:


x


# In[14]:


y=df.iloc[:,-1].values


# In[15]:


y


# # Splitting Dataset into Training set and Test set

# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


# # Logistic Regression model on dataset

# In[17]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)


# In[18]:


pred=classifier.predict(x_test)


# In[19]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,pred)
cm


# In[20]:


acc=accuracy_score(y_test,pred)
acc


# # Decision Tree model on dataset

# In[21]:


from sklearn.tree import DecisionTreeClassifier
dec_classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
dec_classifier.fit(x_train,y_train)


# In[22]:


dec_pred=dec_classifier.predict(x_test)


# In[23]:


cm_dec=confusion_matrix(y_test,dec_pred)
dec_score=accuracy_score(y_test,dec_pred)


# In[24]:


cm_dec


# In[25]:


dec_score


# # Training with Random Forest Classifer on training set

# In[26]:


from sklearn.ensemble import RandomForestClassifier
rand_classifier=RandomForestClassifier(n_estimators=500,criterion='entropy',random_state=0)
rand_classifier.fit(x_train,y_train)


# In[27]:


rand_pred=rand_classifier.predict(x_test)


# In[28]:


cm_rand=confusion_matrix(y_test,rand_pred)
rand_score=accuracy_score(y_test,rand_pred)


# In[29]:


cm_rand


# In[30]:


rand_score


# In[31]:


print("Logistic Regression"+"Accuracy- {:.2%}".format(acc))
print("Decision Tree Classifier"+"Accuracy- {:.2%}".format(dec_score))
print("Random Forest Classifier"+"Accuracy- {:.2%}".format(rand_score))


# In[ ]:

pickle.dump(classifier,open('log_model.pkl','wb'))
pickle.dump(dec_classifier,open('des_model.pkl','wb'))
pickle.dump(rand_classifier,open('Random_forest_model.pkl','wb'))



# In[ ]:




