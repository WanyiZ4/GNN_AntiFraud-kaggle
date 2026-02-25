#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Anti Money Laundering (AML)
# 
# Money laundering is the process used to disguise the source of money or assets derived from criminal activity. Money laundering involves approximately $\$ 590$ billion to $\$ 1.5$ trillion USD per year worldwide. In 2018, $\$46.7$ billion was laundered through Canada with $\$7.4$ billion accounted in British Columbia. Canadian real state is a prime target for money laundering. For banks, it is important to comply the legislation from FINTRAC and OSFI. 
# 
# In terms of the model, it is important to reduce the false positive alarms while having very low tolerance to false negatives. False negatives, are critical because they may be problematic for the bank in terms of legislation and affect the reputation of the bank. It is important to analyze the context of the transactions/users relationships to improve effectiveness of the alarms. 
# 
# 
# 
# # Data: Elliptic Data Set
# 
# The [Elliptic Dataset](https://www.groundai.com/project/anti-money-laundering-in-bitcoin-experimenting-with-graph-convolutional-networks-for-financial-forensics/1) is a graph network of Bitcoin transactions with handcrafted features. All features are constructed using only publicly available information. 
# 
# The Elliptic DataSet maps Bitcoin transactions to real entities in two categories:
# * Licit: exchanges, wallet providers, miners, licit services, etc.
# * Ilicit: scams, malware, terrorist, organization, ransomware, Ponzi shcemes, etc
# 
# A given transaction is licit if the entity that generated it was licit.
# 
# - *Nodes and Edges* 203,769 node transactions and 234,355 directed edge payments flows. 2% are ilicit (Class 1), 21% are licit (Class 2)
# - *Features* Each node has associated 166 features. 94 represent local information (timestep, number of inputs/outputs, transaction fee, output volume and aggregated figures) and 72 features represent aggregated features (obtained by aggregating transaction information such as maximum, minimum, standard deviation, correlation coefficients of neighbor transactions).
# - *Temporal Information* A time step is associated with each node, representing an stimated of the time when the transaction is confirmed. There are 49 distinct timesteps evenly spaced with an interval of 2 weeks.
# 
# ## This notebook
# 1. <a href='#eda'> Exploratory Data Analysis </a>
# 2. <a href='#graphs'> Look into graphs for ilicit and licit transactions </a>
# 3. <a href='#prediction'> Prediction of Ilicit transactions using a simple Random Forest Classifier  </a>
# 
# ## Some Resources
# [Kaggle](https://www.kaggle.com/ellipticco/elliptic-data-set) -
# [Medium](https://medium.com/elliptic/the-elliptic-data-set-opening-up-machine-learning-on-the-blockchain-e0a343d99a14)
# 
# ## Notebook Examples:
# [EDA](https://www.kaggle.com/artgor/elliptic-data-eda)
# 
# [Ilicit transaction detection](https://www.kaggle.com/dhruvrnaik/illicit-transaction-detection)
# 
# 

# ## Import packages

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


# ## Load Data

# In[ ]:


df_classes = pd.read_csv("../input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_classes.csv")
df_edges = pd.read_csv("../input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv")
df_features = pd.read_csv("../input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_features.csv", header=None)


# * Change column names

# In[ ]:


colNames1 = {'0': 'txId', 1: "Time step"}
colNames2 = {str(ii+2): "Local_feature_" + str(ii+1) for ii in range(93)}
colNames3 = {str(ii+95): "Aggregate_feature_" + str(ii+1) for ii in range(72)}

colNames = dict(colNames1, **colNames2, **colNames3 )
colNames = {int(jj): item_kk for jj,item_kk in colNames.items()}


# In[ ]:


df_features = df_features.rename(columns=colNames)
df_features.head()


# * Change class 'unknown' to 3

# In[ ]:


df_classes.loc[df_classes['class'] == 'unknown', 'class'] = 3
print('Shape of classes', df_classes.shape)
print('Shape of edges', df_edges.shape)
print('Shape of features', df_features.shape)


# <a id='eda'></a>
# ## Exploratory Data Analysis (EDA)

# * Quantity of transactions per class

# In[ ]:


group_class = df_classes.groupby('class').count()
plt.barh(['Unknown', 'Ilicit', 'Licit'], group_class['txId'].values, color=['orange', 'r', 'g'] )


# * Number of transactions per Time step

# In[ ]:


group_feature = df_features.groupby('Time step').count()
group_feature['txId'].plot()
plt.title('Number of transactions by Time step')


# * Number of transactions per class and timestep

# In[ ]:


# Merge Class and features
df_class_feature = pd.merge(df_classes, df_features )
df_class_feature.head()


# In[ ]:


group_class_feature = df_class_feature.groupby(['Time step', 'class']).count()
#plt.barh(['Unknown', 'Ilicit', 'Licit'], group_class['txId'].values, color=['orange', 'r', 'g'] )
group_class_feature = group_class_feature['txId'].reset_index().rename(columns={'txId': 'count'})#unstack()
group_class_feature.head()


# In[ ]:


sns.lineplot(x='Time step', y='count', hue='class', data = group_class_feature, palette=['g', 'orange', 'r'] )


# In[ ]:


class1 = group_class_feature[group_class_feature['class'] == '1']
class2 = group_class_feature[group_class_feature['class'] == '2']
class3 = group_class_feature[group_class_feature['class'] == 3 ]

p1 = plt.bar(class3['Time step'], class3['count'], color = 'orange')

p2 = plt.bar(class2['Time step'], class2['count'], color='g',
             bottom=class3['count'])

p3 = plt.bar(class1['Time step'], class1['count'], color='r',
             bottom=np.array(class3['count'])+np.array(class2['count']))

plt.xlabel('Time step')


# <a id='graphs'></a>
# # Graphs
# * Ilicit transactions at time step 20

# In[ ]:


ilicit_ids = df_class_feature.loc[(df_class_feature['Time step'] == 20) & (df_class_feature['class'] == '1'), 'txId']
ilicit_edges = df_edges.loc[df_edges['txId1'].isin(ilicit_ids)]

graph = nx.from_pandas_edgelist(ilicit_edges, source = 'txId1', target = 'txId2', 
                                 create_using = nx.DiGraph())
pos = nx.spring_layout(graph)
nx.draw(graph, with_labels=False, pos=pos)


# In[ ]:


licit_ids = df_class_feature.loc[(df_class_feature['Time step'] == 20) & (df_class_feature['class'] == '2'), 'txId']
licit_edges = df_edges.loc[df_edges['txId1'].isin(licit_ids)]

graph = nx.from_pandas_edgelist(licit_edges, source = 'txId1', target = 'txId2', 
                                 create_using = nx.DiGraph())
pos = nx.spring_layout(graph)
nx.draw(graph, with_labels=False, pos=pos)


# <a id='prediction'></a>
# # Prediction of ilicit transactions
# ## Import packages

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support


# ## Initial Data
# * First we pick the class 1 or 2 transactions only

# In[ ]:


selected_ids = df_class_feature.loc[(df_class_feature['class'] != 3), 'txId']
df_edges_selected = df_edges.loc[df_edges['txId1'].isin(selected_ids)]
df_classes_selected = df_classes.loc[df_classes['txId'].isin(selected_ids)]
df_features_selected = df_features.loc[df_features['txId'].isin(selected_ids)]

# Merge Class and features
df_class_feature_selected = pd.merge(df_classes_selected, df_features_selected )
df_class_feature_selected.head()


# In[ ]:


X = df_class_feature_selected.drop(columns=['txId', 'class', 'Time step']) # drop class, text id and time step
y = df_class_feature_selected[['class']]

# in this case, class 2 corresponds to licit transactions, we chang this to 0 as our interest is the ilicit transactions
y = y['class'].apply(lambda x: 0 if x == '2' else 1 )


# * Split train/test data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=15)


# ## Random Forest Classifier

# In[ ]:


model_RF = RandomForestClassifier().fit(X_train.values,y_train.values)
y_preds = model_RF.predict(X_test.values)

prec,rec,f1,num = precision_recall_fscore_support(y_test.values, y_preds)

print("Random Forest Classifier")
print("Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f"%(prec[1],rec[1],f1[1]))


# ## Predicting 'unknown' transactions
# * Selecting 'unknown' ids

# In[ ]:


unknown_ids = df_class_feature.loc[(df_class_feature['class'] == 3), 'txId']

df_edges_unknown = df_edges.loc[df_edges['txId1'].isin(unknown_ids)]
df_classes_unknown = df_classes.loc[df_classes['txId'].isin(unknown_ids)]
df_features_unknown = df_features.loc[df_features['txId'].isin(unknown_ids)]


# * Predicting Class

# In[ ]:


X_unknown = df_features_unknown.drop(columns=['txId', 'Time step'])
y_unknown_preds = model_RF.predict(X_unknown.values)


# * Replacing 'unknown' class with the predicted one

# In[ ]:


df_classes_unknown = df_classes_unknown.copy()
df_classes_unknown.loc[:, 'class'] = y_unknown_preds


# * Merge class and features

# In[ ]:


df_class_feature_unknown = pd.merge(df_classes_unknown, df_features_unknown )
df_class_feature_unknown.head()


# ## Predicted Graph

# In[ ]:


ilicit_ids = df_class_feature_unknown.loc[(df_class_feature_unknown['Time step'] == 20) & (df_class_feature_unknown['class'] == 1), 'txId']
ilicit_edges = df_edges_unknown.loc[df_edges_unknown['txId1'].isin(ilicit_ids)]

graph = nx.from_pandas_edgelist(ilicit_edges, source = 'txId1', target = 'txId2', 
                                 create_using = nx.DiGraph())
pos = nx.spring_layout(graph)
nx.draw(graph, with_labels=False, pos=pos)


# In[ ]:




