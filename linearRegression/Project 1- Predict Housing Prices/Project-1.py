import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lr
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score as cvs
import helperFunction as hf

# Load the dataset
#column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv('linearRegression\\Project 1- Predict Housing Prices\\HousingData.csv', delimiter=',')#, names=column_names)
print(dataset.head())

print(dataset.describe())
dataset = dataset.drop(['ZN', 'CHAS'], axis=1)
print(dataset.isnull().sum())

#these are the columns that have outliers in the plot the circles are the outliers since they are outside the whiskers of the boxplot ie the 1.5*IQR rule
fig, ax = plt.subplots(ncols=5, nrows=3, figsize=(15, 7))  # 15 subplots total
ax = ax.flatten()
index = 0
for i in dataset.columns:
    sns.boxplot(y=i, data=dataset, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.4)
plt.show()

# since the CRIM column has a lot of outliers we will apply the log1p function to it
#Why log1p? since the CRIM column has a lot of outliers and is skewed to the right, we will apply the log1p function to it. The log1p function is used to transform data that has a lot of zeros or negative values. It is defined as log(1+x) and is useful for transforming data that is skewed to the right. The log1p function is also useful for transforming data that has a lot of zeros or negative values. It is defined as log(1+x) and is useful for transforming data that is skewed to the right.
#The log1p function is also useful for transforming data that has a lot of zeros or negative values. It is defined as log(1+x) and is useful for transforming data that is skewed to the right. The log1p function is also useful for transforming data that has a lot of zeros or negative values. It is defined as log(1+x) and is useful for transforming data that is skewed to the right.
hf.show_plot_before_after(dataset, 'CRIM', np.log1p, plot_type='line')
hf.show_plot_before_after(dataset, 'CRIM', np.log1p, plot_type='box')

# since we are sattisfied with the transformation we will apply it to the dataset
dataset['CRIM'] = np.log1p(dataset['CRIM'])

plt.figure(figsize=(8, 8))
ax = sns.heatmap(dataset.corr(method='pearson').abs(), annot=True, square=True)
plt.show()


# on the basis heatmap we will drop PTRATIO and B since they are not correlated with the target variable MEDV
dataset = dataset.drop(['PTRATIO', 'B'], axis=1)
