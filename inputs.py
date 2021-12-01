import pandas as pd
import numpy as np
import warnings
#Building model
from sklearn import linear_model
from scipy import stats
import statsmodels
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from scipy.stats import bartlett
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
#Data visualisation
import seaborn as sns
sns.set(context='notebook', palette='Spectral', style='darkgrid', font_scale=1.5, color_codes=True)
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

dataset = pd.read_csv('educational_&_environmental_inputs.csv')
dataset.head()

dataset.info()

dataset.describe()

df = pd.DataFrame(dataset)
df

df_x = df[['Human_resources', 'Instructional_materials', 'Physical_facilities', 'Internet_accessibility', 'Transportation', 'Security', 'Students_accomodation', 'Electricity']]
df[['Human_resources', 'Instructional_materials', 'Physical_facilities', 'Internet_accessibility', 'Transportation', 'Security', 'Students_accomodation', 'Electricity']] = (df_x - df_x.mean())/df_x.std()
df

#visualise the data using scatterplot and histogram
sns.set_palette('colorblind')
sns.pairplot(data=df, height=3)

#Visualise the relationship between the features and the response using scatterplots
data = sns.pairplot(df, x_vars = ['Human_resources', 'Instructional_materials', 'Physical_facilities', 'Internet_accessibility', 'Transportation', 'Security', 'Students_accomodation', 'Electricity'], y_vars = ['CGPA'], size=7, aspect=0.7)

#fitting the linear model
x = dataset.drop(['CGPA'], axis=1)
y = dataset.CGPA

sc = StandardScaler()
X = sc.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size=0.25)
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_train)

residuals = y_train.values - y_pred
mean_residuals = np.mean(residuals)
print('Mean of Residuals {}'.format(mean_residuals))

#Using Goldfeld Quandt Test
#Null hypothesis: error terms are homoscedastic & Alternative hypothesis: error terms are heteroscedastic
name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(residuals, X_train)
lzip(name, test)

#Bartlett's test tests the null hypothesis that all input samples are from populations with equal variances
test = bartlett(X_train.flatten(), residuals)
print(test)

#To check normality of error terms/residuals
data = sns.distplot(residuals, kde=True)
data = plt.title('Normality of error terms/residuals')

#to test for autocorrelation
import statsmodels.api as sm
sm.graphics.tsa.plot_acf(residuals, lags=40)
plt.show()

#No perfect multicollinearity
plt.figure(figsize=(20,20))
data = sns.heatmap(dataset.corr(), annot= True, cmap='RdYlGn', square=True)

np.random.seed(0)
df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)

y_train = df_train.pop('CGPA')
x_train = df_train

x_train_lm = sm.add_constant(x_train)
lr_1 = sm.OLS(y_train, x_train_lm).fit()
print(lr_1.summary())

from statsmodels.stats.outliers_influence import variance_inflation_factor
#calculating the VIF value
vif = pd.DataFrame()
vif['Features'] = x_train.columns 
vif['VIF'] = [variance_inflation_factor(x_train.values, i) for i in range(x_train.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

regr = linear_model.LinearRegression()
model = regr.fit(x, y)
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)

#dropping highly correlated variables and insignificant variables
x = x_train.drop(x_train.columns[2], axis=1)

#build a second fitted model
x_train_lm = sm.add_constant(x)
lr_2 = sm.OLS(y_train, x_train_lm).fit()
print(lr_2.summary())

#calculating the VIF value again
vif = pd.DataFrame()
vif['Features'] = x.columns 
vif['VIF'] = [variance_inflation_factor(x_train.values, i) for i in range(x.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

x = x.drop(x_train.columns[7], axis= 1)

#build a third fitted model
x_train_lm = sm.add_constant(x)
lr_3 = sm.OLS(y_train, x_train_lm).fit()
print(lr_3.summary())

x = x.drop(x_train.columns[3], 1)

#build a fourth fitted model
x_train_lm = sm.add_constant(x)
lr_4 = sm.OLS(y_train, x_train_lm).fit()
print(lr_4.summary())

x = x.drop(x_train.columns[4], axis = 1)

#build a fifth fitted model
x_train_lm = sm.add_constant(x)
lr_5 = sm.OLS(y_train, x_train_lm).fit()
print(lr_5.summary())

x = x.drop(x_train.columns[1], 1)

#build a sixth fitted model
x_train_lm = sm.add_constant(x)
lr_7 = sm.OLS(y_train, x_train_lm).fit()
print(lr_7.summary())

x = x.drop(x_train.columns[5], 1)

#build a 7th fitted model
x_train_lm = sm.add_constant(x)
lr_6 = sm.OLS(y_train, x_train_lm).fit()
print(lr_6.summary())

x = x.drop(x_train.columns[6], 1)

#build a 8th fitted model
x_train_lm = sm.add_constant(x)
lr_8 = sm.OLS(y_train, x_train_lm).fit()
print(lr_8.summary())

#calculate VIF for the final value
vif = pd.DataFrame()
vif['Features'] = x.columns 
vif['VIF'] = [variance_inflation_factor(x_train.values, i) for i in range(x.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

x = df.drop(['CGPA', 'Physical_facilities', 'Instructional_materials', 'Internet_accessibility', 'Transportation', 'Students_accomodation', 'Electricity', 'Security'], axis=1)
y = df.CGPA

regr = linear_model.LinearRegression()
model = regr.fit(x, y)
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)

print('R2 score:', lr_8.rsquared)

print('F-statistic:', lr_8.fvalue)
print('Probability of observing value at least as high as F-statistic:', lr_8.f_pvalue)

