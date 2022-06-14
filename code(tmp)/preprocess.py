# -*- encoding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math


### Read data, get relevant features, selecting data to be used
# data = pd.read_csv('C:/Users/NMAIL/Desktop/Looxid/lendingclub_dataset/accepted_2007_to_2018/accepted_2007_to_2018Q4.csv')
data = pd.read_csv('C:/Users/NMAIL/Desktop/Research/counterfactual/data/lendingclub/dataset5/loan.csv')

data = data[data['policy_code'] == 1]
data = data[data['pub_rec'].notna()]
data = data[data['last_fico_range_high'] != 0]
data = data[data['revol_util'].notna()]
data = data[data['fico_range_high'] >= 660]
# data = data[data['annual_inc'] <= 200000]
# data = data[data['verification_status'] == 'Verified']

# print(len(data.index))

# default_categories = ['Default', 'Charged Off', 'Does not meet the credit policy. Status:Charged Off']
# fully_paid_categories = ['Fully Paid', 'Does not meet the credit policy. Status:Fully Paid']

# tmp1 = data[data['loan_status'].isin(default_categories)]

# tmp2 = data[data['loan_status'].isin(fully_paid_categories)]

data = data[['emp_length', 'annual_inc', 'open_acc', 'issue_d', 'earliest_cr_line', 'grade', 'home_ownership', 'purpose', 'addr_state', 'loan_status', 'term']]

data = data.dropna()

# print(len(data.index))

data = data.loc[data['loan_status'].isin(['Fully Paid', 'Charged Off', 'Default', 'Does not meet the credit policy. Status:Fully Paid', 'Does not meet the credit policy. Status:Charged Off'])]
# data = data.loc[~data['purpose'].isin(['other'])]
# data = data.loc[~data['home_ownership'].isin(['OTHER'])]

# print(len(data.index))

# data = data.loc[data['issue_d'].str.contains('2016|2017|2018|2019|2020')]
data = data.loc[data['issue_d'].str.contains('2007|2008|2009|2010|2011')]

cont_features = ['emp_length', 'annual_inc', 'open_acc', 'issue_d', 'earliest_cr_line']
cat_features = ['grade', 'home_ownership', 'purpose', 'addr_state']
# cat_features = ['grade', 'home_ownership', 'purpose']

cont_data = data[cont_features]
cat_data = data[cat_features]
label = data['loan_status']
others = data[['issue_d', 'term']]

dict_emp = {'10+ years':10, '6 years':6, '4 years':4, '< 1 year':0, '2 years':2, '9 years':9, '5 years':5, '3 years':3, '7 years':7, '1 year':1, '8 years':8}
cont_data['emp_length'] = cont_data['emp_length'].map(dict_emp)

end = cont_data['issue_d'].values
start = cont_data['earliest_cr_line'].values
cr_his = []
month_dict = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5,'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}

for i in range(len(end)):
	start_date = start[i].split('-')
	end_date = end[i].split('-')
	month_passed = 12 * (int(end_date[1]) - int(start_date[1])) + month_dict[end_date[0]] - month_dict[start_date[0]]
	cr_his.append(month_passed)

cont_data['cr_history'] = cr_his

cont_data = cont_data[['emp_length', 'annual_inc', 'open_acc', 'cr_history']]

dict_pur = {'credit_card':'debt', 'debt_consolidation':'debt', 'car':'purchase', 'major_purchase':'purchase', 'vacation':'purchase', 'wedding':'purchase', 'medical':'purchase', 'other':'purchase', 'house':'purchase', 'home_improvement':'purchase', 'moving':'purchase', 'renewable_energy':'purchase', 'educational':'educational', 'small_business':'small_business'}

cat_data['purpose'] = cat_data['purpose'].map(dict_pur)

dict_home = {'ANY':'OTHER', 'NONE':'OTHER', 'MORTGAGE':'MORTGAGE', 'RENT':'RENT', 'OWN':'OWN', 'OTHER':'OTHER'}

cat_data['home_ownership'] = cat_data['home_ownership'].map(dict_home)

### Desired class = Fully Paid (1), Contrastive class = Charged Off, Default (0)
dict_label = {'Fully Paid':1, 'Charged Off':0, 'Default':0, 'Does not meet the credit policy. Status:Fully Paid':1, 'Does not meet the credit policy. Status:Charged Off':0}
label = label.map(dict_label)

data = pd.concat([cont_data, cat_data], axis = 1)
data = pd.concat([data, label], axis = 1)
data = pd.concat([data, others], axis = 1)

# for feature in data.columns.tolist():
# 	print(feature, ": ", data[feature].unique())

tmp = data['annual_inc'].to_numpy()

print(np.max(tmp))
print(np.min(tmp))

print(len(data.index))

data.to_csv('C:/Users/NMAIL/Desktop/Research/counterfactual/data/lendingclub/dataset5/2007-2011/data1.csv', index = False)