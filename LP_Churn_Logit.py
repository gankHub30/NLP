import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

# read the data in
churn_df = pd.read_csv("D:/Churn/churn_set_2.csv", header=0, delimiter="\t", \
                   quoting=3 )

# Isolate target data
churn_df['Churn_Indicator'] = np.where(churn_df['Active Indicator'] == 0,1,0)
y = churn_df['Churn_Indicator']

#Examine unique values
print pd.unique(churn_df['Vertical'].values.ravel())
print pd.unique(churn_df['World Sales Region'].values.ravel())

# Change these targets to Integers
churn_df['Vertical'] = churn_df['Vertical'].map({'High Tech': 1, 'Consumer': 2, 'Telco': 3, 'Financial Services':4, 'Unassigned':0})
churn_df['World Sales Region'] = churn_df['World Sales Region'].map({'North America': 1, 'EMEA': 2, 'APAC': 3, 'Latin America':4, 'Unassigned':0})

# We don't need these columns
to_drop = ['Site ID', 'Feature Packs', 'Channel Packs', 'Industry', 'Type of Service', 'Sales Team', 'Segment', 'Account Name', 'Active Indicator', 'Account Owner', 'Account Manager']
churn_feat_space = churn_df.drop(to_drop,axis=1)

# summarize the data
print churn_feat_space.describe()

#Change NaNs to 0
churn_feat_space.fillna(0, inplace=True)

#train columns
train_cols = churn_feat_space.columns[:90]

#number of columns that can be used for model
np.linalg.matrix_rank(churn_feat_space[train_cols].values)

#Linear Regression to check inputs
glm = sm.OLS(churn_feat_space['Churn_Indicator'], churn_feat_space[train_cols], missing='drop')

# fit the model
result = glm.fit()

statistics = pd.Series({'r2': result.rsquared,
                        'adj_r2': result.rsquared_adj})
result_df = pd.DataFrame({'params': result.params,
                          'pvals': result.pvalues,
                          'std': result.bse,
                          'statistics': statistics})

result_df.to_csv("D:/Churn/OLS_results_20151209.csv")

