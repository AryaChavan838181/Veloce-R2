import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score as cvs
import helperFunction as hf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

dataset = pd.read_csv('linearRegression\\Project 1- Predict Housing Prices\\HousingData.csv', delimiter=',')#, names=column_names)
print(dataset.head())

print(dataset.describe())
dataset = dataset.drop(['ZN', 'CHAS'], axis=1)
print(dataset.isnull().sum())


columns_with_nan = ['CRIM', 'INDUS', 'AGE', 'LSTAT']
for col in columns_with_nan:
    dataset[col] = dataset[col].fillna(dataset[col].mean())


print(dataset.isnull().sum())
fig, ax = plt.subplots(ncols=5, nrows=3, figsize=(15, 7))
ax = ax.flatten()
index = 0
for i in dataset.columns:
    sns.boxplot(y=i, data=dataset, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.4)
plt.show()

hf.show_plot_before_after(dataset, 'CRIM', np.log1p, plot_type='line')
hf.show_plot_before_after(dataset, 'CRIM', np.log1p, plot_type='box')

dataset['CRIM'] = np.log1p(dataset['CRIM'])

plt.figure(figsize=(8, 8))
ax = sns.heatmap(dataset.corr(method='pearson').abs(), annot=True, square=True)
plt.show()

dataset = dataset.drop(['PTRATIO', 'B'], axis=1)

X = dataset.drop(['MEDV'], axis=1)
Y = dataset['MEDV']
X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)
Y_pred = lr_model.predict(X_test)
print("---------------------------------------------------------------------OUTPUT------------------------------------------------------------------------")
Y_compare_linear = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
print(Y_compare_linear.head(10))
print("-------------------------------------------------------------------------------------------------------------------------------------------------")
print("Intercept:", lr_model.intercept_)
print("Coefficients:", lr_model.coef_)
print("---------------------------------------------------------------------MAE------------------------------------------------------------------------")
print("MAE:", mean_absolute_error(Y_test, Y_pred))
print("---------------------------------------------------------------------MSE------------------------------------------------------------------------")
print("MSE:", mean_squared_error(Y_test, Y_pred))
print("RMSE: ", np.sqrt(mean_squared_error(Y_test, Y_pred)))
print("---------------------------------------------------------------------R-Squared-------------------------------------------------------------------")
print("R^2 Score:", r2_score(Y_test, Y_pred))
print("-------------------------------------------------------------------------------------------------------------------------------------------------")

