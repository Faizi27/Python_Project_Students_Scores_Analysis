#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


df = pd.read_csv('Expanded_data_with_more_features.csv', encoding = 'unicode_escape')
print (df.head())


# In[10]:


df.describe()


# In[11]:


df.info()


# In[12]:


df.isnull().sum()


# In[13]:


df = df.drop("Unnamed: 0", axis = 1)
print (df.head())


# In[38]:


plt.figure(figsize = (5, 5))
ax = sns.countplot(data = df, x = "Gender")
ax.bar_label(ax.containers[0])
plt.title("Gender Distribution")
plt.show()


# In[18]:


gb = df.groupby("ParentEduc").agg({"MathScore": "mean", "ReadingScore": "mean", "WritingScore":"mean"})
print (gb)


# In[37]:


plt.figure(figsize = (5, 5))
sns.heatmap(gb, annot = True)
plt.title("Relationship between Parents' Education and Students' Scores")
plt.show()


# In[20]:


gb_1 = df.groupby("ParentMaritalStatus").agg({"MathScore": "mean", "ReadingScore": "mean", "WritingScore":"mean"})
print (gb_1)


# In[36]:


plt.figure(figsize = (5, 5))
sns.heatmap(gb_1, annot = True)
plt.title("Relationship between Parents' Marital Status and Students' Scores")
plt.show()


# In[35]:


sns.boxplot(data = df, x = "MathScore")
plt.title("Outliers of Maths Scores")
plt.show()


# In[33]:


sns.boxplot(data = df, x = "ReadingScore")
plt.title("Outliers of Reading Scores")
plt.show()


# In[32]:


sns.boxplot(data = df, x = "WritingScore")
plt.title("Outliers of Writing Scores")
plt.show()


# In[25]:


print(df["EthnicGroup"].unique())


# In[31]:


groupA = df.loc[(df['EthnicGroup'] == "group A")].count()
groupB = df.loc[(df['EthnicGroup'] == "group B")].count()
groupC = df.loc[(df['EthnicGroup'] == "group C")].count()
groupD = df.loc[(df['EthnicGroup'] == "group D")].count()
groupE = df.loc[(df['EthnicGroup'] == "group E")].count()

list_1 = ["group A", "group B", "group C", "group D", "group E"]
m_list = [groupA["EthnicGroup"], groupB["EthnicGroup"], groupC["EthnicGroup"], groupD["EthnicGroup"], groupE["EthnicGroup"]]

plt.pie(m_list, labels = list_1, autopct = "%1.2f%%")
plt.title("Distribution of Ethnic Groups")
plt.show()

