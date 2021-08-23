#!/usr/bin/env python
# coding: utf-8

# # Price Prediction of Used Vehicles

# <p>This is a regression modelling project as my prediction model will be based on Linear Regression.
# I would like to predict the price of used vehicles based on their specifications. The specifications provided in the data set include:
# •	Brand e.g. BMW, Mercedes.
# •	Body e.g. Sedan, Wagon.
# •	Mileage
# •	Engine Volume
# •	Engine Type e.g. Petrol, Diesel.
# •	Registration
# •	Year of manufacture
# •	Model
# </p>

# In[636]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from pandas_profiling import ProfileReport
from sklearn.linear_model import LinearRegression

from statsmodels.stats.outliers_influence import variance_inflation_factor
sns.set()


# In[637]:


from sklearn.preprocessing import StandardScaler


# In[638]:


import warnings
warnings.filterwarnings('ignore')


# In[639]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[718]:


raw_data = pd.read_csv("C:/Users/Dorothy/filesNdata/1.01. Vehicle Specifications.csv")
raw_data.head()


# In[719]:


temp1 = raw_data[['Brand', 'Model']].groupby(['Brand'], as_index = False).count()
temp1 = temp1.set_index('Brand')
temp1


# In[720]:


# Create a new dataframe without the 'Model' column as it is not useful
df = raw_data.drop('Model',axis=1)
df.describe(include='all')


# In[643]:


df.isnull().sum()


# In[721]:


#df['Price'].fillna(value=df['Price'].interpolate(method='linear'),inplace=True)
#df['EngineV'].fillna(value=df['EngineV'].interpolate(method='linear'),inplace=True)
#inplace makes the changes permanent..
#median is not affected by outliers...mean can be affected
#using interpolate...regression method


# In[722]:


# Create a new variable that is made up of the dataframe wihtout the rows with missing values
# data_nomv = data.dropna(axis=0)

df['Price'].fillna(value=df['Price'].median(),inplace=True)
df['EngineV'].fillna(value=df['EngineV'].median(),inplace=True)

df.isnull().sum()


# In[723]:


df.drop_duplicates( keep='first', inplace=True)


# In[724]:


df.info()


# In[648]:


ProfileReport(df)


# In[725]:


df['Year'] = 2020 - df['Year']
df.rename(columns={'Year':'Age'}, inplace=True)


# In[650]:


sns.boxplot(df['Price'],orient='h')


# In[651]:


df['Brand'].value_counts(normalize=False).plot.bar(figsize=(9,7), title='A count of the car brands')


# In[652]:


df['Price'].plot(kind='hist')
plt.show()


# In[653]:


x_axis = df['Brand']
y_axis = df['Price']
plt.figure(figsize=(12,9))
sns.scatterplot(x_axis, y_axis, hue=df['Engine Type'], palette=['g','r','c','m'])
plt.title('Brand, Price and Engine Type')


# In[654]:


x_axis = df['Brand']
y_axis = df['Mileage']
plt.figure(figsize=(12,9))
sns.scatterplot(x_axis, y_axis, hue=df['Engine Type'], palette=['g','r','c','m'])
plt.title('Brand, Mileage and Engine Type')


# In[655]:


x_axis = df['Brand']
y_axis = df['EngineV']
plt.figure(figsize=(12,9))
sns.scatterplot(x_axis, y_axis, hue=df['Engine Type'], palette=['g','r','c','m'])
plt.title('Brand, Engine Volume and Engine Type')


# In[656]:


x_axis = df['Age']
y_axis = df['Price']
plt.figure(figsize=(12,9))
sns.scatterplot(x_axis, y_axis, hue=df['Engine Type'], palette=['g','r','c','m'])
plt.title('Age, Price and Engine Type')


# In[657]:


x_axis = df['Brand']
y_axis = df['Price']
plt.figure(figsize=(12,9))
sns.scatterplot(x_axis, y_axis, hue=df['Engine Type'], palette=['g','r','c','m'])
plt.title('Brand, Price and Engine Type')


# In[658]:


sns.pairplot(data=df)


# In[659]:


plt.figure(figsize=(12,6))
sns.barplot(data=df,x='Brand',y='Price',hue='Engine Type')
plt.title('Scatter plot for Brand Price')
plt.show()


# In[660]:


plt.figure(figsize=(12,6))
sns.barplot(data=df,x='Brand',y='Price',hue='Registration')
plt.title('Scatter plot for Brand Price')
plt.show()


# In[661]:


sns.boxplot(x=df['Price'])


# In[662]:


sns.boxplot(df['EngineV'])


# In[663]:


sns.boxplot(df['Mileage'])


# In[664]:


sns.boxplot(df['Age'],orient='h')


# In[726]:


q1 = np.percentile(df['Age'],25)
q3 = np.percentile(df['Age'],75)

iqr = q3-q1
minimum = q1 - 1.5*iqr
max_age = q3 + 1.5*iqr
print('Q1: %s Q2: %s min: %s max: %s'%(q1,q3,minimum,max_age))


df[df['Age']>max_age]


# In[727]:


q1 = np.percentile(df['EngineV'],25)
q3 = np.percentile(df['EngineV'],75)

iqr = q3-q1
minimum = q1 - 1.5*iqr
max_engv = q3 + 1.5*iqr
print('Q1: %s Q2: %s min: %s max: %s'%(q1,q3,minimum,max_engv))


df[df['EngineV']>max_engv]


# In[728]:


q1 = np.percentile(df['Mileage'],25)
q3 = np.percentile(df['Mileage'],75)

iqr = q3-q1
minimum = q1 - 1.5*iqr
max_mile = q3 + 1.5*iqr
print('Q1: %s Q2: %s min: %s max: %s'%(q1,q3,minimum,max_mile))


df[df['Mileage']>max_mile]


# In[729]:


q1 = np.percentile(df['Price'],25)
q3 = np.percentile(df['Price'],75)

iqr = q3-q1
minimum = q1 - 1.5*iqr
max_price = q3 + 1.5*iqr
print('Q1: %s Q2: %s min: %s max: %s'%(q1,q3,minimum,max_price))


df[df['Price']>max_price]


# In[730]:



df = df[df['Age']<max_age]
df = df[df['Price']<max_price]
df = df[df['Mileage']<max_mile]
df = df[df['EngineV']<max_engv]


# In[731]:


df


# In[732]:


# price_mileage = df['Price'][df['Price']>max_price][df['Mileage']>max_mile]
# indice1 = price_mileage.index
# list1 = indice1.tolist()

# price_engv = df['Price'][df['Price']>max_price][df['EngineV']>max_engv]
# indice2 = price_engv.index
# list2 = indice2.tolist()

# mile_age = df['Mileage'][df['Mileage']>max_mile][df['Age']>max_age]
# indice3 = mile_age.index
# list3 = indice3.tolist()
# lists =list1 +list2 +list3


# In[733]:


# df.iloc[lists]


# In[734]:


# df2=df.copy()
# df2.drop(lists,inplace=True)


# In[735]:


df.reset_index(drop=True,inplace=True)


# In[680]:


# df2['Price'][df2['Price']>max_price2]=df2['Price'].median()
# df2['Mileage'][df2['Mileage']>max_mile2]=df2['Mileage'].median()
# df2['EngineV'][df2['EngineV']>max_engv2]=df2['EngineV'].median()


# In[681]:


x_axis = df['Brand']
y_axis = df['Price']
plt.figure(figsize=(12,9))
sns.scatterplot(x_axis, y_axis, hue=df['Engine Type'], palette=['g','r','c','m'])
plt.title('Brand, Mileage and Engine Type')


# In[682]:


sns.pairplot(data=df)
# Check for linearity in the continuous variables
# Price, Age and Mileage


# In[736]:


df['Price'].max()


# In[684]:


# Check for linearity in the continuous variables
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,3))
ax1.scatter(df['Age'],df['Price'])
ax1.set_title('Price and Age')
ax2.scatter(df['EngineV'],df['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(df['Mileage'],df['Price'])
ax3.set_title('Price and Mileage')


# In[737]:


#transforming 'Price'to a linear variable using logarithm because of its exponential distribution
log_price = np.log(df['Price'])
df['log_price'] = log_price


# In[686]:


sns.distplot(df['Price'])


# In[687]:


# Viewing the Price PDF once again to notice its exponential distribution
sns.distplot(df['log_price'])


# In[688]:


# Check for linearity in the continuous variables
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,3))
ax1.scatter(df['Age'],df['log_price'])
ax1.set_title('Price and Age')
ax2.scatter(df['EngineV'],df['log_price'])
ax2.set_title('Price and EngineV')
ax3.scatter(df['Mileage'],df['log_price'])
ax3.set_title('Price and Mileage')


# In[689]:


#df2.drop(['log_price'],axis=1,inplace=True)


# In[738]:


#define the features you want to check for multicorrelation
variables = df2[['Mileage','Age','EngineV']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['features'] = variables.columns
vif.round(1)


# <p>VIF Values between 1 and 5 are considered acceptable.
# <br>The feature 'Year' has the highest VIF hence it more likely correlated to the other variables and I shall now drop it from the dataframe</p>

# In[691]:


#df2.drop(['Age'],axis=1,inplace=True)


# In[739]:


# Spotting all categorical variables and creating dummies
# If we have N categorical variables, we have to create N-1 dummies so if all other dummies are zero, then we shall conclude that it is the category that isn't a dummy
df_final = pd.get_dummies(df, drop_first=True)

df_final.head()


# In[708]:


df_final.shape


# In[740]:


#variable to hold all the column names and arrange them as I want with log_price coming first
cols = ['log_price','Price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']


# In[741]:


df_final = df_final[cols]


# In[742]:


df_final


# In[743]:


df_final.drop(['Price','Registration_yes'],axis=1,inplace=True)


# In[744]:


# df_final2=df_final.copy() 


# In[745]:


#scale the data using StandardScaler
#create an instance of the StandardScaler class
#fit the inputs to the scaler instance
# We can scale the dummies as it has no impact on their predicitve power

# scaler = StandardScaler()
# scaler.fit(df_final2)
# df_scaled = scaler.transform(df_final2)


# In[746]:


# df_scaled


# In[747]:


X = df_final.iloc[:,[1,2]].values
targets = df_final.iloc[:,0].values
scaler = StandardScaler()
scaler.fit(X)
features = scaler.transform(X)
features


# In[748]:


scaler = StandardScaler()
scaler.fit(df_final.iloc[:,[1,2]])
featurez = scaler.transform(df_final.iloc[:,[1,2]])
featurez


# In[749]:


featurez.shape


# In[750]:


X2.shape


# In[751]:


df_trimmed = df_final.drop(['Mileage','EngineV'],axis=1,inplace=True)
X2 = df_final.iloc[:,1:].values
targets = df_final.iloc[:,0].values


# In[752]:


X2


# In[753]:


features2 = np.concatenate((featurez,X2),axis=1)
features2


# In[754]:


targets


# In[755]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(features,targets,test_size=0.2,random_state=1)


# In[756]:


from sklearn.linear_model import LinearRegression
#definition
model_linear = LinearRegression(normalize=True)
#fitting the data
model_linear.fit(X_train, y_train)
#prediction
y_pred = model_linear.predict(X_test)
y_pred


# In[757]:


from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_score, recall_score


# In[758]:


#model validation
print(f'MAE: {mean_absolute_error(y_test,y_pred)}')
print(f'MSE: {mean_squared_error(y_test,y_pred)}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test,y_pred))}')

print(f"R2_score: {r2_score(y_test,y_pred)}")


# In[759]:


plt.scatter(y_test,y_pred)


# In[760]:


from sklearn.linear_model import SGDRegressor
from statsmodels.stats.anova import anova_lm


# In[761]:


# Using Stochastic Gradient Descent
#sgd_reg = SGDRegressor(penalty="l2", max_iter=1000, tol=1e-3)#penalty l2 indicates that you want
#l2 is ridge regression
#l1 is lasso regression
#Difference is that ridge adds squared magnitude of coefficient as penalty term to the loss function
#Lasso(least absolute shrinkage and Selection Operator) adds "absolute value of magnitude" of coeffiecient as penalty term to the loss function
#Lasso shrinks the less important feature's coefficient to zero thus, removing some feature altogether, works well with feature selection
#SGD to add a regularization term to the cost function equal to half the square of the l2 norm of the weight vector


# 
# Stochastic Gradient Descent Regressor
# 
# The gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule(learning rate)
# 
# The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1 or a combination of both (Elastic Net)
# 
# tol : float or None, optional (default=1e-3) The stopping criterion. If it is not None, the iterations will stop when (loss > best_loss - tol) for n_iter_no_change consecutive epochs.
# 
# max_iter...The maximum number of passes over the training data (aka epochs).
# 
# eta0... The initial learning rate for the 'constant'
# 

# In[762]:


sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1, tol=1e-3)
sgd_reg.fit(X_train, y_train.ravel())#.ravel() returns a flatenned array


# In[763]:


sgd_reg.intercept_, sgd_reg.coef_
y_predict_sdg=sgd_reg.predict(X_test)
y_predict_sdg


# In[764]:


mse2=mean_squared_error(y_predict_sdg,y_test)
mae2=mean_absolute_error(y_predict_sdg,y_test)
print(f"Mean Squared Error: {mse2}")
print(f"Mean Absolute Eror: {mae2}")


# In[765]:


plt.scatter(y_test,y_predict_sdg)


# In[766]:



X_train2, X_test2, y_train2,y_test2 = train_test_split(features2,targets,test_size=0.2,random_state=1)

#definition
model_linear2 = LinearRegression(normalize=True)
#fitting the data
model_linear2.fit(X_train2, y_train2)
#prediction
y_pred2 = model_linear2.predict(X_test2)
y_pred2


# In[767]:


#model validation
print(f'MAE: {mean_absolute_error(y_test2,y_pred2)}')
print(f'MSE: {mean_squared_error(y_test2,y_pred2)}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test2,y_pred2))}')

print(f"R2_score: {r2_score(y_test2,y_pred2)}")


# In[768]:


plt.scatter(y_test2,y_pred2)


# In[769]:


sgd_reg2 = SGDRegressor(max_iter=50, penalty=None, eta0=0.1, tol=1e-3)
sgd_reg2.fit(X_train2, y_train2.ravel())#.ravel() returns a flatenned array


# In[770]:


sgd_reg2.intercept_, sgd_reg2.coef_
y_predict2_sdg=sgd_reg2.predict(X_test2)
y_predict2_sdg


# In[771]:


mse2=mean_squared_error(y_predict2_sdg,y_test2)
mae2=mean_absolute_error(y_predict2_sdg,y_test2)
print(f"Mean Squared Error: {mse2}")
print(f"Mean Absolute Eror: {mae2}")
print(f'RMSE: {np.sqrt(mean_squared_error(y_test2,y_predict2_sdg))}')
print(f"R2_score: {r2_score(y_test2,y_predict2_sdg)}")


# In[772]:


plt.scatter(y_test,y_predict_sdg)


# In[773]:


# Calculating the Bias
model_linear2.intercept_


# In[774]:


#Calculating the Weights of the features
model_linear2.coef_


# In[775]:


colz = ['Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol']


# In[776]:


# create a summary table of the features and their weights
reg_summary = pd.DataFrame(data=colz, columns=['Features'])
reg_summary['Weights'] = model_linear2.coef_
reg_summary


# In[777]:


df_pf = pd.DataFrame(data=np.exp(y_pred2), columns=['Prediction'])
df_pf.head()


# In[778]:


df_pf['Target'] = np.exp(y_test2)
df_pf.head()


# In[779]:


# Create a column for the residuals
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
#Create a column for the absolute residual percentage
df_pf['Residual%'] = np.absolute(df_pf['Residual'])/df_pf['Target']*100
df_pf.head()


# In[780]:


# sorting df_pf by difference in percentages using the sort values method 
df_pf.sort_values(by=['Residual%'])

