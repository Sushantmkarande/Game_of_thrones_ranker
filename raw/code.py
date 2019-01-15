
# coding: utf-8

# In[250]:


train = pd.read_csv('train.csv')


# In[251]:


test = pd.read_csv('test.csv')
test_original=test.copy()


# In[252]:


# All project packages imported at the start

# Project packages
import pandas as pd
import numpy as np

# Visualisations
import matplotlib.pyplot as plt 
import seaborn as sns

# Statistics
from scipy import stats
from scipy.stats import norm, skew
from statistics import mode

# Machine Learning
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso, Ridge, RidgeCV, ElasticNet
# import xgboost as xgb
# import lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge

# from catboost import Pool, CatBoostRegressor, cv

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    

    
    


# In[253]:


# train = train.head(1000)


# In[254]:


train.info()


# In[255]:


test.info()


# In[256]:


train.head()


# In[257]:


train_ID = train['soldierId']
test_ID = test['soldierId']

train.drop("soldierId", axis = 1, inplace = True)
test.drop("soldierId", axis = 1, inplace = True)



train.drop("shipId", axis = 1, inplace = True)
test.drop("shipId", axis = 1, inplace = True)
train.drop("attackId", axis = 1, inplace = True)
test.drop("attackId", axis = 1, inplace = True)


# In[258]:


# fig = plt.gcf()
# fig.set_size_inches(18.5, 10.5)
# fig.savefig('test2png.png', dpi=100)

# axes = plt.gca()
# # axes.set_xlim([xmin,xmax])
# axes.set_ylim([0.0,1.0])
# sns.regplot(x=train['assists'], y=train['bestSoldierPerc'], fit_reg=True)
# plt.show()


# In[259]:


# fig = plt.gcf()
# fig.set_size_inches(18.5, 10.5)
# fig.savefig('test2png.png', dpi=100)

# axes = plt.gca()
# # axes.set_xlim([xmin,xmax])
# axes.set_ylim([0.0,1.0])
# sns.regplot(x=train['greekFireItems'], y=train['bestSoldierPerc'], fit_reg=True)
# plt.show()


# In[260]:


train.head()


# In[286]:


train.isnull().sum()


# In[288]:


test.isnull().sum()


# In[262]:


train['knockedOutSoldiers'].fillna(train['knockedOutSoldiers'].mode()[0], inplace=True)


# In[263]:


train['horseRideDistance'].fillna(train['horseRideDistance'].mode()[0], inplace=True)
train['respectEarned'].fillna(train['respectEarned'].mode()[0], inplace=True)


# In[290]:


# print(train.shape)
test.shape


# In[289]:


x = train.drop(['bestSoldierPerc',],1)
y = train.bestSoldierPerc
x.shape


# In[281]:


from sklearn import cross_validation


# In[282]:


x_train, x_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size =0.2)


# In[283]:


clf = LinearRegression()
clf.fit(x_train,y_train)
accuracy = clf.score(x_test,y_test)


# In[284]:


accuracy


# In[291]:


pred_test = clf.predict(test)


# In[292]:


submission=pd.read_csv("Sample_Submission.csv")


# In[293]:


submission['bestSoldierPerc']=pred_test
submission['soldierId']=test_original['soldierId']


# In[294]:


pd.DataFrame(submission, columns=['soldierId','bestSoldierPerc']).to_csv('sub.csv')

