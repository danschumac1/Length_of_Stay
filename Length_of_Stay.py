"""
Created on 04/26/2024

@author: Dan Schumacher
"""

#endregion
#region # IMPORTS
# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd
import datetime as dt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from sklearn.linear_model import LinearRegression
import itertools
from tqdm import tqdm


#endregion
#region # LOAD DATA
# =============================================================================
# LOAD DATA
# =============================================================================
df_los = pd.read_csv('./data/los_cleaned.csv')

# MAKE TARGET VARIABEL
# For each row, we will subtract the row’s ALOS from its GM-LOS and multiply by $1,000.
df_los.loc[:, 'LOSDiscrepancyCost'] = -(df_los['LOS'] - df_los['GM-LOS'] * 1000)
df_los = df_los.dropna(subset=['LOSDiscrepancyCost'])

# Turn ageGroup into median of range
df_los

#endregion
#region # DESCRIPTIVE STATISTICS
# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================
df_los.drop(['patientID','dischargeQTR','month'], axis = 1).describe()
# on avg people are admitted earlier in the week (@$@ check what 0 is)

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