#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns 


# In[2]:


boston=pd.read_csv('train.csv')
y = boston['medv']
# Drop specified labels from rows or columns. Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
x = boston.drop('medv', axis = 1)
print("Boston housing dataset has {} data points with {} variables each.".format(*boston.shape))


# In[3]:


# Minimum price of the data
minimum_price = np.amin(y)

# Maximum price of the data
maximum_price = np.amax(y)

# Mean price of the data
mean_price = np.mean(y)

# Median price of the data
median_price = np.median(y)

# Standard deviation of prices of the data
std_price = np.std(y)

# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${}".format(minimum_price)) 
print("Maximum price: ${}".format(maximum_price))
print("Mean price: ${}".format(mean_price))
print("Median price ${}".format(median_price))
print("Standard deviation of prices: ${}".format(std_price))


# In[4]:


boston.head()


# In[5]:


# to check whether there is any missing value in given data
boston.isnull().sum()


# In[6]:


# setting the plot size for all plots
sns.set(rc={'figure.figsize':(11,8)})
sns.distplot(boston['medv'], bins=30)
# bins define the number of bars in histogram
plt.show()


# We see that the values of MEDV are distributed normally with few outliers.
# 
# A correlation matrix is a type of matrix that measures the linear relationships between the variables. The correlation matrix can be formed by using the corr function from the pandas dataframe library. We will use the heatmap function from the seaborn library to plot the correlation matrix.

# In[7]:


correlation_matrix = boston.corr()
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)


# In[8]:


boston[[col for col in boston]].corr()


# In[9]:


plt.figure(figsize=(20, 5))

features = ['lstat', 'rm']
target = boston['medv']

for i, col in enumerate(features):
    # subplot(nrows, ncols, plot_number)
    plt.subplot(1, len(features) , i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')  # o represents circle
    plt.xlabel(col)
    plt.ylabel('medv')


# In[10]:


# This describes the mean, standard deviation, minimum, and maximum for each column
boston.describe().T


# In[11]:


relevant_features=['crim', 'rm', 'lstat', 'medv']
relevant_features


# In[12]:


# alpha= define points opacity
sns.pairplot(boston[relevant_features],plot_kws={'alpha':0.6},diag_kws={'bins':30})


# In[13]:


fig,ax=plt.subplots(1,2)
# regplot calculates the best fit line by automatically minimizing the ordinary least squares error function
sns.regplot('rm','medv',boston,ax=ax[0],scatter_kws={'alpha':0.6})
sns.regplot('lstat','medv',boston,ax=ax[1],scatter_kws={'alpha':0.6})


# In[16]:


X = pd.DataFrame(np.c_[boston['lstat'], boston['rm']], columns = ['lstat','rm'])
Y = boston['medv']


# In[17]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[18]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


# In[20]:


# model evaluation for training set
from sklearn.metrics import r2_score
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[ ]:




