#!/usr/bin/env python
# coding: utf-8

# In[106]:


import pandas as pd 
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import warnings
warnings.simplefilter('ignore')


# In[2]:


fraud=pd.read_csv('/Users/angelawu/Documents/DS take home challenge/Identifying fradulent activities/Fraud.csv')


# In[3]:


ip_country=pd.read_csv('/Users/angelawu/Documents/DS take home challenge/Identifying fradulent activities/IpAddress_to_Country.csv')


# In[4]:


fraud.head()


# In[5]:


ip_country.head()


# In[6]:


fraud.info()


# In[7]:


ip_country.info()


# ### Data Explore Analysis

# #### Add country column to the fraud table 

# In[8]:


country=[]
for ind,row in fraud.iterrows():
    tmp=ip_country[(ip_country['lower_bound_ip_address']<=row['ip_address']) & (ip_country['upper_bound_ip_address']>=row['ip_address'])]['country']
    if len(tmp)==1:
        country.append(tmp.values[0])
    else:
        country.append('NA')
fraud['country']=country
        


# In[9]:


fraud.head()


# #### Feature engineering

# 1. time difference between sign-up time and purchase time 
# 2. device_id: if multiple user ids using the same device could be an indicator of fraudulent. 
# 3. ip_address: many different users having the same ip address could be an indicator of fraudulent.
# 4. datetime feature engineering

# In[18]:


# time difference 
df=fraud.copy()
df['signup_time']=pd.to_datetime(df['signup_time'])
df['purchase_time']=pd.to_datetime(df['purchase_time'])
df['tdiff']=(df['purchase_time']-df['purchase_time'])/np.timedelta64(1, 's')


# In[25]:


# number of users using the same device 
df['device_user_ct']=df.groupby('device_id')['user_id'].transform('count')


# In[37]:


# number of users' using a given ip address
df['ip_ct']=df.groupby('ip_address')['user_id'].transform('count')


# In[42]:


sum(df['device_user_ct']==1)/len(df)


# In[45]:


sum(df['ip_ct']==1)/len(df)


# 87.2% of users match one device and 94.4% users use one ip address. 

# In[73]:


# day of the week and the day
df['signup_day']=df['signup_time'].dt.dayofweek
df['signup_week']=df['signup_time'].dt.week

df['purchase_day']=df['purchase_time'].dt.dayofweek
df['purchase_week']=df['purchase_time'].dt.week


# In[74]:


data=df[['purchase_value','source','browser','sex','age','class','country','device_user_ct','ip_ct','signup_day','signup_week','purchase_day','purchase_week']]


# In[75]:


data


# #### Categorical encoding

# In[76]:


# One hot encoding

data=pd.get_dummies(data,columns=['source','browser','sex'],drop_first=True)


# In[77]:


# frequency encoding for country column 

country_freq=data['country'].value_counts(normalize=True)
data['country_ec']=data['country'].map(lambda x:country_freq[x])


# In[78]:


data=data.drop(['country'],axis=1)


# In[79]:


data['class'].value_counts(normalize=True)


# The data is imbalanced. 

# ### Splitting data into train and test dataset

# In[82]:


X=data.drop(['class'],axis=1)
y=data['class']


# In[85]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=42,stratify=y)


# ### Balancing data with over-sampling (SMOTE)

# In[90]:


smo=SMOTE()
X_smo,y_smo=smo.fit_resample(X_train,y_train)
sns.countplot(y_smo)


# ### Logistic regression model

# In[98]:


lg=LogisticRegression()
lg.fit(X_smo,y_smo)
y_pred=lg.predict(X_test)


# In[100]:


confusion_matrix=confusion_matrix(y_test,y_pred)
confusion_matrix


# In[103]:


roc_auc_score(y_test,y_pred)


# ### Random forest model

# In[114]:


rf=RandomForestClassifier()
rf.fit(X_smo,y_smo)
y_pred_rf=rf.predict(X_test)


# In[116]:


roc_auc_score(y_test,y_pred_rf)


# ### What kinds of users are more likely to be classiﬁed as at risk?

# In[121]:


imp=pd.DataFrame(rf.feature_importances_,index=X_smo.columns,columns=['importance'])


# In[126]:


imp=imp.sort_values('importance',ascending=True)


# In[127]:


imp.plot(kind='barh')


# ### From a product perspective, how would you use this model?

# If predicted fraud probability < X, the user has the normal experience (the high majority should fall here)
# If X <= predicted fraud probability < Z (so the user is at risk, but not too much), you can create an additional verification step, like verify your phone number via a code sent by SMS or log in via Facebook.
# If predicted fraud probability >= Z (so here is really likely the user is trying to commit a fraud), you can tell the user his session has been put on hold. 
