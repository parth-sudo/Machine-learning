#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# importing imp modules


# In[2]:


df = pd.read_csv("pokemre.csv")
df.head()


# In[3]:


Y=df.Name
features=['Total','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Generation','Legendary']
X=df[features]


# In[4]:


train_X, val_X, train_y, val_y = train_test_split(X, Y, test_size=0.3,random_state=66)


# In[5]:


iowa_model = RandomForestClassifier(random_state=1, criterion= "entropy")


# In[6]:


iowa_model.fit(train_X, train_y)


# In[7]:


val_prediction = iowa_model.predict(val_X)
print(val_prediction)


# In[12]:


print("Enter the following info")
Total1=input("Enter total-:")
HP1=input("Enter HP-:")
Attack1=input("Enter Attack-:")
Defense1=input("Enter defence-:")
SpAtk1=input("Enter sp attack-:")
SpDef1=input("Enter sp defence-:")
Speed1=input("Enter speed")
Generation1=input("Enter generation(from 1 to 6) -:")
Legendary1=input("Enter legendary status(if yes press 1 or else 0) -:")
pre=[[Total1,HP1,Attack1, Defense1,SpAtk1,SpDef1,Speed1, Generation1,Legendary1]]
print(*iowa_model.predict(pre))


# In[ ]:




