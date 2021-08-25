import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import streamlit as st
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_agg import RendererAgg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

sns.set()

_lock = RendererAgg.lock

st.set_page_config(layout="wide")

st.title('Price Prediction of Used Vehicles')
st.write(
    '''

This is a regression modelling project based on Linear Regression.
I would like to predict the price of used vehicles based on their specifications. The specifications provided in the data set include:
•	Brand e.g. BMW, Mercedes.
•	Body e.g. Sedan, Wagon.
•	Mileage
•	Engine Volume
•	Engine Type e.g. Petrol, Diesel.
•	Registration
•	Year of manufacture
•	Model
    '''
)

raw_data = pd.read_csv("1.01. Vehicle Specifications.csv")
st.write('''## EXPLORATORY DATA ANALYSIS''')
st.write('')
st.write('The dataset')
st.write(raw_data.sample(n=7))

st.write('')

temp1 = raw_data[['Brand', 'Model']].groupby(['Brand'], as_index = False).count()
temp1 = temp1.set_index('Brand')

st.write('The Model column is dropped as it is not useful')
df = raw_data.drop('Model',axis=1)
st.write(df.describe(include='all'))

#df['Price'].fillna(value=df['Price'].interpolate(method='linear'),inplace=True)
#df['EngineV'].fillna(value=df['EngineV'].interpolate(method='linear'),inplace=True)
#inplace makes the changes permanent..
#median is not affected by outliers...mean can be affected
#using interpolate...regression method

# Create a new variable that is made up of the dataframe wihtout the rows with missing values
# data_nomv = data.dropna(axis=0)

df['Price'].fillna(value=df['Price'].median(),inplace=True)
df['EngineV'].fillna(value=df['EngineV'].median(),inplace=True)
df.drop_duplicates( keep='first', inplace=True)

df['Year'] = 2020 - df['Year']
df.rename(columns={'Year':'Age'}, inplace=True)

st.write('')
row1_1, row1_2, row1_3 = st.beta_columns((1, 2, 2))
with row1_1, _lock:
    st.write('The models')
    st.write(temp1)

with row1_2, _lock:
    fig = plt.figure()
    ax = fig.subplots()
    sns.histplot(df['Price'], kde_kws={'clip': (0.0, 5.0)}, ax=ax, kde=True)
    ax.set_ylabel('Totals')
    ax.set_xlabel('Price')
    st.pyplot(fig)

with row1_3, _lock:
    fig = plt.figure(figsize=(9,7))
    ax = fig.subplots()
    brand_count_df = pd.DataFrame(raw_data['Brand'].value_counts(normalize=False))
    sns.barplot(x=brand_count_df.index,y=brand_count_df['Brand'], color="red")
    ax.set_ylabel('Totals')
    ax.set_xlabel('Brand')
    st.pyplot(fig)

st.write('')
row2_1, row2_2 = st.beta_columns((1, 1))
with row2_1, _lock:
    st.write('**Brand, Mileage and Price**')
    fig = plt.figure(figsize=(10,8))
    ax = fig.subplots()
    sns.scatterplot(df['Brand'],  df['Price'], hue=df['Engine Type'], palette=['g','r','c','m'])
    ax.set_xlabel('Price')
    ax.set_ylabel('Density')
    st.pyplot(fig)

with row2_2, _lock:
    st.write('**Brand, Engine Volume and Engine Type**')
    fig = plt.figure(figsize=(10,8))
    ax = fig.subplots()
    sns.scatterplot(df['Brand'],  df['EngineV'], hue=df['Engine Type'], palette=['g','r','c','m'])
    ax.set_xlabel('Price')
    ax.set_ylabel('Density')
    st.pyplot(fig)

st.write('')
row3_1, row3_2 = st.beta_columns((1, 1))
with row3_1, _lock:
    st.write('**Age, Price and Engine Type**')
    fig = plt.figure(figsize=(10,8))
    ax = fig.subplots()
    sns.scatterplot(df['Age'],  df['Price'], hue=df['Engine Type'], palette=['g','r','c','m'])
    ax.set_xlabel('Price')
    ax.set_ylabel('Density')
    st.pyplot(fig)

with row3_2, _lock:
    st.write('**Brand, Engine Volume and Engine Type**')
    fig = plt.figure(figsize=(10,8))
    ax = fig.subplots()
    sns.scatterplot(df['Brand'],  df['Price'], hue=df['Engine Type'], palette=['g','r','c','m'])
    ax.set_xlabel('Price')
    ax.set_ylabel('Density')
    st.pyplot(fig)

row4_1, row4_2 = st.beta_columns((1, 1))
with row4_1, _lock:
    st.write('**Age, Price and Engine Type barplot**')
    fig = plt.figure(figsize=(10,8))
    ax = fig.subplots()
    sns.barplot(data=df,x='Brand',y='Price',hue='Engine Type')
    ax.set_xlabel('Price')
    ax.set_ylabel('Density')
    st.pyplot(fig)

with row4_2, _lock:
    st.write('**Brand, price and Registration barplot**')
    fig = plt.figure(figsize=(10,8))
    ax = fig.subplots()
    sns.barplot(data=df,x='Brand',y='Price',hue='Registration')
    ax.set_xlabel('Price')
    ax.set_ylabel('Density')
    st.pyplot(fig)


q1 = np.percentile(df['Age'],25)
q3 = np.percentile(df['Age'],75)

iqr = q3-q1
minimum = q1 - 1.5*iqr
max_age = q3 + 1.5*iqr


# df[df['Age']>max_age]

q1 = np.percentile(df['EngineV'],25)
q3 = np.percentile(df['EngineV'],75)

iqr = q3-q1
minimum = q1 - 1.5*iqr
max_engv = q3 + 1.5*iqr
# df[df['EngineV']>max_engv]

q1 = np.percentile(df['Mileage'],25)
q3 = np.percentile(df['Mileage'],75)

iqr = q3-q1
minimum = q1 - 1.5*iqr
max_mile = q3 + 1.5*iqr

# df[df['Mileage']>max_mile]

q1 = np.percentile(df['Price'],25)
q3 = np.percentile(df['Price'],75)

iqr = q3-q1
minimum = q1 - 1.5*iqr
max_price = q3 + 1.5*iqr

# df[df['Price']>max_price]

df = df[df['Age']<max_age]
df = df[df['Price']<max_price]
df = df[df['Mileage']<max_mile]
df = df[df['EngineV']<max_engv]

price_mileage = df['Price'][df['Price']>max_price][df['Mileage']>max_mile]
indice1 = price_mileage.index
list1 = indice1.tolist()

price_engv = df['Price'][df['Price']>max_price][df['EngineV']>max_engv]
indice2 = price_engv.index
list2 = indice2.tolist()

mile_age = df['Mileage'][df['Mileage']>max_mile][df['Age']>max_age]
indice3 = mile_age.index
list3 = indice3.tolist()
lists =list1 + list2 + list3

df.reset_index(drop=True,inplace=True)

st.write('After some data cleaning')
row5_1, row5_2 = st.beta_columns((1, 1))
with row5_1, _lock:
    st.write('**Brand, Mileage and Engine Type**')
    fig = plt.figure(figsize=(10,8))
    ax = fig.subplots()
    sns.scatterplot(df['Brand'], df['Price'], hue=df['Engine Type'], palette=['g','r','c','m'])
    ax.set_xlabel('Price')
    ax.set_ylabel('Density')
    st.pyplot(fig)

with row5_2, _lock:
    st.write('**Pairplot with the clean Data**')
    fig = sns.pairplot(data=df)
    st.pyplot(fig)

st.write('Using the pairplot above and the scatter plots below to check for linearity in the continous variables')

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,3))
ax1.scatter(df['Age'],df['Price'], color="green")
ax1.set_title('Price and Age')
ax2.scatter(df['EngineV'],df['Price'], color="green")
ax2.set_title('Price and EngineV')
ax3.scatter(df['Mileage'],df['Price'], color="green")
ax3.set_title('Price and Mileage')
st.pyplot(f)

st.write('I decided to transform Price to a linear variable using a log function because of its exponential distribution')
df['log_price'] = np.log(df['Price'])

row6_1, row6_2 = st.beta_columns((1, 1))
with row6_1, _lock:
    st.write('**Distribution plot with price**')
    fig = plt.figure()
    sns.distplot(df['Price'])
    st.pyplot(fig)

with row6_2, _lock:
    st.write('**Distribution plot with log price**')
    fig = plt.figure()
    sns.distplot(df['log_price'])
    st.pyplot(fig)

st.write('Defining the features I want to check for multicorrelation')
variables = df[['Mileage','Age','EngineV']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['features'] = variables.columns
st.write(vif.round(1))
st.write('''
VIF Values between 1 and 5 are considered acceptable
''')

# Spotting all categorical variables and creating dummies
# If we have N categorical variables, we have to create N-1 dummies so if all other dummies are zero, then we shall conclude that it is the category that isn't a dummy
df_final = pd.get_dummies(df, drop_first=True)

st.write(f"After cleaning and making neccesary data manipulations, the new dataset now has {df_final.shape[0]} rows and {df_final.shape[1]} columns")
st.write('**The new dataset**')
st.write(df_final.sample(n=8))

#variable to hold all the column names and arrange them as I want with log_price coming first
cols = ['log_price','Price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']

df_final = df_final[cols]

df_final.drop(['Price','Registration_yes'],axis=1,inplace=True)

# df_final2=df_final.copy() 

#scale the data using StandardScaler
#create an instance of the StandardScaler class
#fit the inputs to the scaler instance
# We can scale the dummies as it has no impact on their predicitve power

# scaler = StandardScaler()
# scaler.fit(df_final2)
# df_scaled = scaler.transform(df_final2)

st.write('''
I transformed the data using StandardScaler by creating an instance of the StandardScaler class and fitting the inputs to the scaler instance
''')

X = df_final.iloc[:,[1,2]].values
targets = df_final.iloc[:,0].values
scaler = StandardScaler()
scaler.fit(X)
features = scaler.transform(X)

# df_trimmed = df_final.drop(['Mileage','EngineV'],axis=1,inplace=True)
X2 = df_final.iloc[:,1:].values
targets = df_final.iloc[:,0].values

# features2 = np.concatenate((features,X2),axis=1)
features2 = df_final[['Mileage', 'EngineV']]
X_train, X_test, y_train,y_test = train_test_split(features,targets,test_size=0.2,random_state=1)
X_train2, X_test2, y_train2,y_test2 = train_test_split(features2,targets,test_size=0.2,random_state=1)

# X is Mileage and EngineV categorical features
# X2 is the rest of the categorical features

st.write('''Using Stochastic Gradient Descent
- sgd_reg = SGDRegressor(penalty="l2", max_iter=1000, tol=1e-3)#penalty l2 indicates that you want
- l2 is ridge regression
- l1 is lasso regression
- Difference is that ridge adds squared magnitude of coefficient as penalty term to the loss function
- Lasso(least absolute shrinkage and Selection Operator) adds "absolute value of magnitude" of coeffiecient as penalty term to the loss function
- Lasso shrinks the less important feature's coefficient to zero thus, removing some feature altogether, works well with feature selection
- SGD to add a regularization term to the cost function equal to half the square of the l2 norm of the weight vector
''')

st.write('''
Stochastic Gradient Descent Regressor

- The gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule(learning rate)

- The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1 or a combination of both (Elastic Net)

- tol : float or None, optional (default=1e-3) The stopping criterion. If it is not None, the iterations will stop when (loss > best_loss - tol) for n_iter_no_change consecutive epochs.

- max_iter...The maximum number of passes over the training data (aka epochs).

- eta0... The initial learning rate for the 'constant'
''')

res=sm.OLS(targets,features2).fit()
st.write(res.summary())
st.write('Checking the effect of dropping a single observation')
ols_results2 = sm.OLS(targets[:14], features2[:14]).fit()
st.write(tuple([i for i in (ols_results2.params - res.params)/res.params*100]))
# st.write("Percentage change is %f" % tuple([i for i in (ols_results2.params - res.params)/res.params*100]))

#Calculating the Weights of the features

model_linear = LinearRegression(normalize=True)
model_linear.fit(X_train, y_train)
y_pred = model_linear.predict(X_test)
mae = ("{:.3f}%".format(100*mean_absolute_error(y_test2,y_pred)))
mse = ("{:.3f}%".format(100*mean_squared_error(y_test2,y_pred)))
rmse = ("{:.3f}%".format(100*np.sqrt(mean_squared_error(y_test2,y_pred))))
st.write(f'The bias {model_linear.intercept_}')
r2_score("{:.3f}%".format(100*r2_score(y_test,y_pred)))
errors = abs(y_pred - y_test2)
accuracy = 100 - np.mean(100*(errors/y_test2))
acc = (round(accuracy, 2), '%.')

sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1, tol=1e-3)
sgd_reg.fit(X_train, y_train)#.ravel() returns a flatenned array
y_predict_sdg=sgd_reg.predict(X_test)
mae2 = ("{:.3f}%".format(100*mean_absolute_error(y_predict_sdg,y_test)))
mse2 = ("{:.3f}%".format(100*mean_squared_error(y_predict_sdg,y_test)))
rmse2 = ("{:.3f}%".format(100*np.sqrt(mean_squared_error(y_predict_sdg,y_test))))
st.write(f'The bias {sgd_reg.intercept_}')
r2_score2 = ("{:.3f}%".format(100*r2_score(y_test,y_predict_sdg)))
errors2 = abs(y_predict_sdg - y_test2)
accuracy2 = 100 - np.mean(100*(errors2/y_test2))
acc2 = (round(accuracy2, 2), '%.')
    
poly_features = PolynomialFeatures(degree=2, include_bias = False)#generate a polynomial matrix containing all the polnomial features of the specified degree
X_poly = poly_features.fit_transform(X_train2)#fits then transforms the training set X into a polynomial set
X_polytest = poly_features.fit_transform(X_test2)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y_train2)
y_predict3 = pol_reg.predict(X_polytest)
mae3 = ("{:.3f}%".format(100*mean_absolute_error(y_test2,y_predict3)))
mse3 = ("{:.3f}%".format(100*mean_squared_error(y_test2,y_predict3)))
rmse3 = ("{:.3f}%".format(100*np.sqrt(mean_squared_error(y_test2,y_predict3))))
st.write(f'The bias {pol_reg.intercept_}')
r2_score3 = ("{:.3f}%".format(100*r2_score(y_test2,y_predict3)))
errors3 = abs(y_predict3 - y_test2)
accuracy3 = 100 - np.mean(100*(errors/y_test2))
acc3 = (round(accuracy3, 2), '%.')

ridge_reg = Ridge(alpha=1, solver="cholesky", penalty="l2")
ridge_reg.fit(X_train2, y_train2)
y_predridge=ridge_reg.predict(X_test2)
mae4 = ("Mean Absolute Error: {:.3f}%".format(100*mean_absolute_error(y_test2,y_predridge)))
mse4 = ("Mean Square Error: {:.3f}%".format(100*mean_squared_error(y_test2,y_predridge)))
rmse4 = ("RMSE: {:.3f}%".format(100*np.sqrt(mean_squared_error(y_test2,y_predridge))))
st.write(f'The bias {ridge_reg.intercept_}')
r2_score4 = ("R2_score: {:.3f}%".format(100*r2_score(y_test2,y_predridge)))
mse4=mean_squared_error(y_predridge,y_test2)
mae4=mean_squared_error(y_predridge,y_test2)
errors4 = abs(y_predict4 - y_test2)
accuracy4 = 100 - np.mean(100*(errors/y_test2))
acc4 = (round(accuracy4, 2), '%.')

colz = ['Mileage', 'EngineV']

lasso_reg = Lasso(alpha = 0.1)
lasso_reg.fit(X_train2, y_train2)
y_predlasso = lasso_reg.predict(X_test2)
mae5 = ("Mean Absolute Eror: {:.3f}%".format(100*mean_absolute_error(y_predlasso,y_test2)))
mse5 = ("Mean Squared Error: {:.3f}%".format(100*mean_squared_error(y_predlasso,y_test2)))
rmse5 = ("RMSE: {:.3f}%".format(100*np.sqrt(mean_squared_error(y_predlasso,y_test2))))
st.write(f'The bias {lasso_reg.intercept_}')
r2_score5 = ("R2_score: {:.3f}%".format(100*r2_score(y_test2,y_predlasso)))
errors5 = abs(y_predlasso - y_test2)
accuracy5 = 100 - np.mean(100*(errors5/y_test2))
acc5 = (round(accuracy5, 2), '%.')


elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X_train2, y_train2)
y_predict5 = elastic_net.predict(X_test2)
mae6 = ("{:.3f}%".format(100*mean_absolute_error(y_predict5,y_test2)))
mse6 = ("{:.3f}%".format(100*mean_squared_error(y_predict5,y_test2)))
rmse6 = ("{:.3f}%".format(100*np.sqrt(mean_squared_error(y_predict5,y_test2))))
st.write(f'The bias {elastic_net.intercept_}')
r2_score6 = ("{:.3f}%".format(100*r2_score(y_test2,y_predict5)))
errors6 = abs(y_predict5 - y_test2)
accuracy6 = 100 - np.mean(100*(errors6/y_test2))
acc6 = (round(accuracy6, 2), '%.')


decisionregressor=DecisionTreeRegressor()
decisionregressor.fit(X_train2,y_train2)
st.write('Decision Tree')
y_pred_dec=decisionregressor.predict(X_test2)
score=r2_score(y_test2,y_pred_dec)
mae7 =("{:.3f}%".format(100*mean_absolute_error(y_pred_dec,y_test2)))
mse7 = ("{:.3f}%".format(100*mean_squared_error(y_pred_dec,y_test2)))
rmse7 = ("{:.3f}%".format(100*np.sqrt(mean_squared_error(y_pred_dec,y_test2))))
# st.write(f'The bias {elastic_net.intercept_}')
r2_score7 = ("{:.3f}%".format(100*r2_score(y_test2,y_pred_dec)))
errors7 = abs(y_pred_dec - y_test2)
accuracy7 = 100 - np.mean(100*(errors7/y_test2))
acc7 = (round(accuracy7, 2), '%.')

#Instantiating with 1000 decision trees
st.write('Random Forest')
rf = RandomForestRegressor(n_estimators=1000, random_state = 42)
rf.fit(X_train2, y_train2)
predictions = rf.predict(X_test2)
mae8 = ("{:.3f}%".format(100*mean_absolute_error(predictions,y_test2)))
mse8 = ("{:.3f}%".format(100*mean_squared_error(predictions,y_test2)))
rmse8 = ("{:.3f}%".format(100*np.sqrt(mean_squared_error(predictions,y_test2))))
# st.write(f'The bias {elastic_net.intercept_}')
r2_score8 = ("{:.3f}%".format(100*r2_score(y_test2,predictions)))
errors8 = abs(predictions - y_test2)
accuracy8 = 100 - np.mean(100*(errors8/y_test2))
acc8 = (round(accuracy8, 2), '%.')

errors = abs(predictions - y_test2)
st.write('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
accuracy = 100 - np.mean(100*(errors/y_test2))
st.write('Accuracy:', round(accuracy, 2), '%.')
importances = list(rf.feature_importances_)

feature_list = colz
#List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
#Sorting the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
#Print out the feature and importances
[st.write('Variable: {:20} Importance:{}'.format(*pair)) for pair in feature_importances]

data1 = [mae, mae2, mae3, mae4, mae5, mae6, mae7, mae8]
data2 = [mse, mse2, mse3, mse4, mse5, mse6, mse7, mae8]
data3 = [rmse, rmse2, rmse3, rmse4, rmse5, rmse6, rmse7, rmse8]
data4 = [r2_score,r2_score1, r2_score2, r2_score3,r2_score4, r2_score5, r2_score6,r2_score7, r2_score8]
data5 = [acc, acc2, acc3, acc4, acc5,acc6, acc7, acc8]


index = ['linear_regression','SDG_regression','Polynomial_Regression','Ridge_regression','Lasso_regression','ElasticNet_regression', 'Decision Tree', 'Random Forest]
errord = pd.DataFrame({'absol_error':data1, 'sqrd_error':data2, 'root_mean_sq_error':data3, 'r2_score':data4, 'accuracy':data5},index = index)
errord.sort_values(by=['accuracy'])
st.write(errord)


# create a summary table of the features and their weights
reg_summary = pd.DataFrame(data=colz, columns=['Features'])
reg_summary['Weights'] = model_linear2.coef_
# with row
df_pf = pd.DataFrame(data=np.exp(y_pred2), columns=['Prediction'])

df_pf['Target'] = np.exp(y_test2)

# Create a column for the residuals
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
#Create a column for the absolute residual percentage
df_pf['Residual%'] = np.absolute(df_pf['Residual'])/df_pf['Target']*100
df_pf.head()

df_pf.sort_values(by=['Residual%'])
# sorting df_pf by difference in percentages using the sort values method 

row9_1, row9_2 = st.beta_columns((1,1))
with row9_1, _lock:
    st.subheader('A summary table of the features and their weights')
    reg_summary
with row9_2, _lock:
    st.subheader('A summary table of the predictions and residuals(absolute differences)')
    df_pf
