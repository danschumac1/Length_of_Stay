#!/usr/bin/env python
# coding: utf-8

# # Imports 

# In[31]:


import pandas as pd
from pandas.api.types import CategoricalDtype

import datetime as dt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from sklearn.linear_model import LinearRegression

import itertools
from tqdm import tqdm

import sys
sys.path.append('./functions')
from functions import (
    logistic_regression_diagnostic_plots,
    diagnostic_plots,
    calculate_vif,
    remove_high_vif_features,
    evaluate,
    plot_sensitivity_specificity,
    aic_scorer,
    select_model_by_aic,
    view_logistic_regression_coefficients,
    calculate_cooks_distance
)


# # Load the Data

# In[32]:


df_los = pd.read_csv('./data/los_cleaned.csv')


# ### Develop the target variable

# In[33]:


# For each row, we will subtract the row’s ALOS from its GM-LOS and multiply by $1,000.
df_los.loc[:, 'LOSDiscrepancyCost'] = -(df_los['LOS'] - df_los['GM-LOS'] * 1000)
df_los = df_los.dropna(subset=['LOSDiscrepancyCost'])


# #### Data cleaning. 

# In[34]:


# Here we can see that there are only 4 observations of this category (which doesn't allow us to treat this as a ordinal var)
# therfor, let us drop it 
df_los = df_los.drop(df_los[df_los['riskOfMortality'] == '0-Ungroupable'].index)

# These are weird NA values. Let us explicitly make them NAs
df_los['principalProcedure'] = df_los['principalProcedure'].replace('-', np.nan)


# In[35]:


len(df_los)


# In[36]:


df_los.head()


# ## Get descriptive statistics

# In[37]:


import pandas as pd

# Initialize a dictionary to store the values
data_dict = {
    'Metric': [],
    'Value': []
}

# Number of cases
num_cases = len(df_los)
data_dict['Metric'].append('Number of Cases')
data_dict['Value'].append(num_cases)

# Average GM-LOS
avg_gm_los = df_los['GM-LOS'].mean()
data_dict['Metric'].append('Average GM-LOS')
data_dict['Value'].append(avg_gm_los)

# Average CMI
avg_cmi = df_los['caseMixIndex'].mean()
data_dict['Metric'].append('Average CMI')
data_dict['Value'].append(avg_cmi)

# Split dataframe into 2 hospitals
df_mch = df_los[df_los['hospital'] == df_los['hospital'].unique()[0]]
df_sh = df_los[df_los['hospital'] == df_los['hospital'].unique()[1]]

# Average LOS for each hospital
# Medical Center Hospital
avg_mch_los = df_mch['LOS'].mean()
# South Hospital
avg_sh_los = df_sh['LOS'].mean()

data_dict['Metric'].append('Medical Center Hospital LOS')
data_dict['Value'].append(avg_mch_los)

data_dict['Metric'].append('South Hospital LOS')
data_dict['Value'].append(avg_sh_los)

# Create a DataFrame from the dictionary
metrics_df = pd.DataFrame(data_dict)

# Print the DataFrame
metrics_df


# ### How much money would each hospital save if they could get their LOSDiscrepancyCost to 0 for every patient?

# In[38]:


mch_neg_values = pd.DataFrame()
mch_neg_values['LOSDiscrepancyCost'] = df_mch[df_mch['LOSDiscrepancyCost'] > 0]['LOSDiscrepancyCost']
mch_money_saved = mch_neg_values.sum()[0]

sh_neg_values = pd.DataFrame()
sh_neg_values['LOSDiscrepancyCost'] = df_sh[df_sh['LOSDiscrepancyCost'] > 0]['LOSDiscrepancyCost']
sh_money_saved = sh_neg_values.sum()[0]

print('MCH Money Saved:', f'${mch_money_saved:,.2f}')
print('SH Money Saved:', f'${sh_money_saved:,.2f}')


# ### What product lines have the greatest opportunity for improvement in terms of patient days?
# Group by (product line) -> avg. sort by highest to lowest cost.
# ### Visualize with bar charts.
# 

# In[39]:


# isolate vars that are negative

pl_improvement = df_los.groupby('productLine').sum(numeric_only=True).sort_values(by = 'LOSDiscrepancyCost', ascending = False)['LOSDiscrepancyCost']
pl_improvement = pd.DataFrame(pl_improvement)
pl_improvement.reset_index(inplace=True)
pl_improvement.head()


# In[40]:


plt.figure(figsize=(12,6))

# Function to format y-axis labels to display numerical values in dollars with commas
def format_y_tick_labels(value, pos):
    return f'${value:,.0f}'

# Create the barplot
sns.barplot(x='productLine', y='LOSDiscrepancyCost', data=pl_improvement.head(10))

# Rotate x-axis labels
plt.xticks(rotation=45)

# Format y-axis labels using the formatter function
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_tick_labels))


plt.title('Product Lines with the biggest potential for improvement (LOS)')
plt.xlabel('Product Line')
plt.ylabel('Estimated Money Lossed')

# Show the plot
plt.show()


# ### Within the product line with the greatest opportunity for improvement (identified above), which DRG has the greatest opportunity for improvement?  Answer the questions: What is this diagnosis? Why is it difficult for hospitals to manage?
# - Within the product line identified above, group by (DRG) -> avg. sort by highest to lowest cost.
#  - Visualize with bar charts.
# 

# In[41]:


# isolate vars that are negative
gen_med = df_los[df_los['productLine'] == 'General Medice']
drg_improvement = df_los.groupby(['APR DRG']).sum(numeric_only=True).sort_values(by = 'LOSDiscrepancyCost', ascending = False)['LOSDiscrepancyCost']
drg_improvement = pd.DataFrame(drg_improvement)
drg_improvement.reset_index(inplace=True)
drg_improvement.head()


# In[42]:


# @$@ clean this up later

plt.figure(figsize=(12,6))

# Create the barplot
sns.barplot(x='APR DRG', y='LOSDiscrepancyCost', data=drg_improvement.head(10))

# Rotate x-axis labels
plt.xticks(rotation=45)

# Format y-axis labels using the formatter function
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_tick_labels))


plt.title('Product Lines with the biggest potential for improvement (LOS)')
plt.xlabel('Product Line')
plt.ylabel('Estimated Money Lossed')

# Show the plot
plt.show()


# ### Build a predictive model focused on interpretability that uses the patient demographic data to see which group of patients is expected to have the highest variance in length of stay. Build a predictive model that focuses on predictive power rather than interpretability that allows hospitals to more accurately predict LOS deviation.
# - Linear Regression for interpretable model
# - Try other types of models, for predictive purposes. Support vector machines, random forest etc.

# In[43]:


# toy example 
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

# Sample DataFrame
data = {
    'sex': ['Male', 'Female', 'Female', 'Male', 'Male'],
    'hospital': ['Hospital A', 'Hospital B', 'Hospital C', 'Hospital A', 'Hospital B'],
    'age': [25, 30, 35, 40, 45]
}

df = pd.DataFrame(data)

# Convert columns to categorical
df['sex'] = df['sex'].astype('category')
df['hospital'] = df['hospital'].astype('category')

# Using patsy to create a design matrix
formula = 'age ~ sex + hospital'
y, X = dmatrices(formula, df, return_type='dataframe')

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)


# In[44]:


df_los['dischargeQTR']


# In[15]:


# regress_me = df_los.drop(
#     ["patientID", # IRRELEVANT BC EACH IS UNIQUE
#     "LOS", # WOULD BE TOO CORELATED WITH TARGET
#     "LOSGroupName", # WOULD BE TOO CORELATED WITH TARGET
#     "totalCharge", # CAN'T LOOK INTO THE FUTURE
#     "dischargeStatus", # CAN'T LOOK INTO THE FUTURE
#     "principleDiagnosisCode", 
#     "principalProcedureCode", 
#     "altProductLine1", # KISS
#     "altProductLine1SUB", # KISS
#     'AM-LOS', # WOULD BE TOO CORELATED WITH TARGET
#     'GM-LOS', # WOULD BE TOO CORELATED WITH TARGET
#      'hospital', # LATER SPLIT AND DO REGRESSION ON EACH SO BOTH, HOS1, HOS2,
#      'APR DRG', # KISS MAYBE ADD LATER,
#      'dischargeQTR', # will be a less specific month
#      'principalDiagnosis'# KISS ADD LATER
#     ]
#     , axis=1)

# categorify = [
#     'admitType',
#     'month',
#     'admittedDOW',
#     'ageGroup',
#     'sex',
#     'race',
#     'admitType',
#     'ethnicity',
# #     'principalDiagnosis',
#     'productLine',
#     'payCode',
# ]


# cat_type = CategoricalDtype(categories=['1-Minor', '2-Moderate', '3-Major', '4-Extreme'], ordered=True)
# regress_me['riskOfMortality'] = regress_me['riskOfMortality'].astype(cat_type)
# regress_me['severity'] = regress_me['severity'].astype(cat_type)

# for cat in categorify:
#     regress_me[cat] = regress_me[cat].astype('category')


# In[67]:


regress_me = df_los.drop(
    ["patientID", # IRRELEVANT BC EACH IS UNIQUE
    "LOS", # WOULD BE TOO CORELATED WITH TARGET
    "LOSGroupName", # WOULD BE TOO CORELATED WITH TARGET
    "totalCharge", # CAN'T LOOK INTO THE FUTURE
    "dischargeStatus", # CAN'T LOOK INTO THE FUTURE
    "principleDiagnosisCode", 
    "principalProcedureCode", 
    "altProductLine1", # KISS
    "altProductLine1SUB", # KISS
    'AM-LOS', # WOULD BE TOO CORELATED WITH TARGET
    'GM-LOS', # WOULD BE TOO CORELATED WITH TARGET
     'hospital', # LATER SPLIT AND DO REGRESSION ON EACH SO BOTH, HOS1, HOS2,
     'APR DRG', # KISS MAYBE ADD LATER,
     'dischargeQTR', # will be a less specific month
     'principalDiagnosis'# KISS ADD LATER
    ], axis=1)

regress_me = pd.get_dummies(regress_me, drop_first=True)


# In[ ]:





# In[68]:


import pandas as pd
import numpy as np

def filter_highly_correlated(df, threshold=0.8):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(to_drop, axis=1)

# Assuming df is your DataFrame and 'MEDV' is your target variable
filtered_df = filter_highly_correlated(regress_me.drop(columns='LOSDiscrepancyCost'))


# In[69]:


from concurrent.futures import ThreadPoolExecutor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm
import pandas as pd

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    
    # Define a function to compute VIF for a given index
    def compute_vif(index):
        return variance_inflation_factor(X.values, index)
    
    # Use ThreadPoolExecutor to parallelize the VIF calculation
    with ThreadPoolExecutor() as executor:
        vifs = list(tqdm(executor.map(compute_vif, range(X.shape[1])), total=X.shape[1], desc='Calculating VIF'))
    
    vif_data['VIF'] = vifs
    return vif_data

# Example usage
vif_df = calculate_vif(filtered_df)
print(vif_df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


for cat in categorify:
    print(f'{cat}:',len(regress_me[cat].unique()))


# In[22]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'regress_me' is your DataFrame prepared earlier
# Select which categories to encode
categories_to_encode = ['admitType', 'sex', 'race', 'ethnicity', 'admittedDOW', 'month', 'payCode']

# Apply one-hot encoding with drop_first to avoid multicollinearity
regress_me_encoded = pd.get_dummies(regress_me, columns=categories_to_encode, drop_first=True)

# Include 'ageGroup' with ordinal encoding, assuming it has a natural order:
age_order = sorted(regress_me['ageGroup'].unique())  # Ensure the order is correct
regress_me_encoded['ageGroup'] = regress_me['ageGroup'].astype('category').cat.reorder_categories(age_order).cat.codes

# Compute the correlation matrix
correlation_matrix = regress_me_encoded.corr()

# Generate a heatmap to visualize the correlation matrix
plt.figure(figsize=(15, 12))  # Adjust size based on the number of variables
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix Including Categorical Variables')
plt.show()


# In[ ]:


import seaborn as sns


# In[181]:


regress_me.columns


# In[143]:


# Check for constant columns
constant_columns = [col for col in X.columns if X[col].std() == 0]
print("Constant columns:", constant_columns)

# Check for perfectly correlated columns
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
perfectly_correlated_columns = [column for column in upper_tri.columns if any(upper_tri[column] == 1)]
print("Perfectly correlated columns:", perfectly_correlated_columns)


# In[183]:


from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from patsy import dmatrices
from joblib import Parallel, delayed
from tqdm import tqdm

# Formula creation and design matrix
target = 'LOSDiscrepancyCost'
independent_vars = [col for col in regress_me.columns if col != target]
formula = f"{target} ~ " + " + ".join(independent_vars)
y, X = dmatrices(formula, regress_me, return_type='dataframe')

# Scale the features excluding the 'Intercept'
features = X.columns.drop('Intercept')
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X[features]), columns=features)

# Identify and remove low variance columns
low_variance_cols = [col for col in X_scaled.columns if X_scaled[col].var() < 1e-10]
X_clean = X_scaled.drop(columns=low_variance_cols)

# Identify highly correlated pairs
cor_matrix = X_clean.corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
X_reduced = X_clean.drop(columns=to_drop)

# Check variance again after removing highly correlated features
low_variance = [col for col in X_reduced.columns if X_reduced[col].var() < 1e-10]
X_reduced = X_reduced.drop(columns=low_variance)

# Function to calculate VIF for a single feature
def calculate_vif(index, data):
    try:
        return variance_inflation_factor(data.values, index)
    except Exception as e:
        return f"Error: {str(e)}"

# Calculate VIF using the cleaned and scaled data in parallel
results = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(calculate_vif)(i, X_reduced) for i in tqdm(range(X_reduced.shape[1]), desc="Calculating VIF"))

# Collect results
vif_data = pd.DataFrame({
    'feature': X_reduced.columns,
    'VIF': results
})

print(vif_data)


# In[153]:


from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from patsy import dmatrices

# Formula creation and design matrix
target = 'LOSDiscrepancyCost'
independent_vars = [col for col in regress_me.columns if col != target]
formula = f"{target} ~ " + " + ".join(independent_vars)
y, X = dmatrices(formula, regress_me, return_type='dataframe')

# Scale the features excluding the 'Intercept'
features = X.columns.drop('Intercept')
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X[features]), columns=features)

# Identify and remove low variance columns
low_variance_cols = [col for col in X_scaled.columns if X_scaled[col].var() < 1e-10]
X_clean = X_scaled.drop(columns=low_variance_cols)

# Identify highly correlated pairs
cor_matrix = X_clean.corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
X_reduced = X_clean.drop(columns=to_drop)


# Check variance again after removing highly correlated features
low_variance = [col for col in X_reduced.columns if X_reduced[col].var() < 1e-10]
X_reduced = X_reduced.drop(columns=low_variance)
# Calculate VIF using the cleaned and scaled data
vif_data = pd.DataFrame(columns=['feature', 'VIF'])
for i, col in enumerate(X_reduced.columns):
    try:
        vif = variance_inflation_factor(X_reduced.values, i)
        vif_data = vif_data.append({'feature': col, 'VIF': vif}, ignore_index=True)
    except Exception as e:
        print(f"Error calculating VIF for {col}: {str(e)}")

print(vif_data)


# In[154]:


vif_data


# In[147]:


from sklearn.preprocessing import StandardScaler


# Using patsy to create a design matrix
target = 'LOSDiscrepancyCost'
independent_vars = [col for col in regress_me.columns if col != target]
formula = f"{target} ~ " + " + ".join(independent_vars)
y, X = dmatrices(formula, regress_me, return_type='dataframe')

# Calculate VIF, excluding the 'Intercept' from the calculation
vif_data = pd.DataFrame()
# Filter out 'Intercept' from the columns to calculate VIF
features = X.columns.drop('Intercept')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[features])
# Drop columns with very low variance
low_variance_cols = [col for col in features if X[col].var() < 1e-10]
X_clean = X.drop(columns=low_variance_cols)

vif_data["feature"] = features
vif_data["VIF"] = [variance_inflation_factor(X[features].values, i) for i in range(len(features))]

print(vif_data)


# In[ ]:


# AIC Selection


# In[ ]:


# Build final model


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


cols_to_categorize = [
    'month',
    'admitType',
    'dischargeStatus',
    'admittedDOW',
    'hospital',
    'ageGroup',
    'sex',
    'race',
    'principalProcedure',
    'productLine',
    'altProductLine1',
    'altProductLine1SUB',
    'payCode'
]

categorical_cols = []
for col in cols_to_categorize:
    df_los[f'{col}_cat'] = df_los[col].astype('category')
    categorical_cols.append(f'{col}_cat')


# In[12]:


regression_ready = df_los.copy()
regression_ready = regression_ready.drop(categorical_cols, axis = 1)


regression_ready =regression_ready.drop(
    ['LOS',
     'APR_DRG',
     'AM-LOS',
     'patientID',
     'dischargeQTR',
     'GM-LOS',
     'caseMixIndex',
     'principleDiagnosisCode',
     'principalDiagnosis',
     'principalProcedureCode',
     'principalProcedure',
     'LOSGroupName'],
    axis = 1
)


# Lets take out categorical variables with too many categories...

var_list = []
var_list.append('LOSDiscrepancyCost')
for col in regression_ready.columns:
    if len(regression_ready[col].unique()) < 26:
        var_list.append(col)

regression_ready = regression_ready[var_list]
regression_ready =pd.get_dummies(regression_ready, drop_first = True)


# In[13]:


corr_mar = regression_ready.corr(numeric_only=True)
# plt.figure(figsize=(8,6))
# sns.heatmap(
#     corr_mar,
#     cmap = 'coolwarm'
# )
# plt.show()


# In[14]:


# getting rid of high vif
X = regression_ready.drop('LOSDiscrepancyCost', axis = 1)
y = regression_ready['LOSDiscrepancyCost']
kept, removed = remove_high_vif_features(X = X, y = y, vif_threshold =  10)


# In[15]:


removed


# In[16]:


corr_mar = kept.corr(numeric_only=True)


# plt.figure(figsize=(8,6))
# sns.heatmap(
#     corr_mar,
#     cmap = 'coolwarm'
# )
# plt.show()


# In[17]:


kept.columns


# In[18]:


# FIT LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = kept.drop(['severity_2-Moderate', 'severity_3-Major','severity_4-Extreme'], axis = 1)
y = regression_ready['LOSDiscrepancyCost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
lr_reg = LinearRegression()
lr_reg = lr_reg.fit(X_train, y_train)

lr_preds = lr_reg.predict(X_test)
mean_squared_error(lr_preds, y_test)


# In[ ]:


# As categorical


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


from sklearn.ensemble import RandomForestRegressor

# FIT RANDOM FOREST
# get features needed
max_features = X_train.shape[1]
tree_count   = 1000

# set up model
rf_reg = RandomForestRegressor(
    max_features=max_features,
    random_state=27,
    n_estimators=tree_count
)

# fit model
rf_reg.fit(X_train, y_train)
# make predictions
rf_preds = rf_reg.predict(X_test)
# get mse
rf_mse = mean_squared_error(y_test, rf_preds)


# In[20]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'X_train' is a pandas DataFrame and you have already defined 'rf_reg' and fit it to your data

# Create a DataFrame with feature names and their importance scores
plot_df = pd.DataFrame({
    'feature': X_train.columns,  # Use the column names from the DataFrame
    'importance': rf_reg.feature_importances_  # Feature importances from the RandomForestRegressor
})

# Plotting feature importances
plt.figure(figsize=(10, 10))
sns.barplot(x='importance', y='feature', data=plot_df.sort_values('importance', ascending=False).head(10))
plt.xticks(rotation=90)
plt.title('Feature Imortance')
plt.show()


# In[ ]:


# %load_ext autoreload
# %autoreload 2

# feature_sets = []
# feature_names = [col for col in X.columns if col != 'LOSDiscrepancyCost']

# # Total number of iterations
# total_iterations = sum(1 for L in range(1, 5) for _ in itertools.combinations(feature_names, L))

# # Create tqdm instance
# pbar = tqdm(total=total_iterations, desc="Feature Set Combinations")

# for L in range(1, 5):  # Change '7' to a different number if you want more features in the combinations
#     for subset in itertools.combinations(feature_names, L):
#         feature_sets.append(list(subset))
#         pbar.update(1)  # Update progress bar

# pbar.close()  # Close progress bar after loop completes

# best_aic, best_features = select_model_by_aic(X, y, feature_sets)
# print(f"Best AIC Score: {best_aic}")
# print(f"Best Feature Set: {best_features}")


# In[ ]:


from sklearn.linear_model import LinearRegression
print(LinearRegression)


# In[ ]:


import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(kept, y, test_size=0.33, random_state=42)

# Fit the OLS model using the training data
model = sm.OLS(y_train, X_train)
results = model.fit()

# Print the summary of the regression results
print(results.summary())
results.summary(alpha = .05)


# In[ ]:


logistic_regression_diagnostic_plots,
diagnostic_plots,
calculate_vif,
remove_high_vif_features,
evaluate,
plot_sensitivity_specificity,
aic_scorer,
select_model_by_aic,
view_logistic_regression_coefficients,
calculate_cooks_distance


# In[ ]:


# FIT LINEAR REGRESSION
from sklearn.linear_model import LinearRegression

lr_reg = LinearRegression()
lr_reg.fit(X, y)
final_week_pred = rf_reg.predict(X_test)

