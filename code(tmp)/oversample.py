import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import pickle


# data = pd.read_csv('C:/Users/NMAIL/Desktop/Research/counterfactual/data/lendingclub/dataset5/2007-2020/data1.csv').to_numpy()
data = pd.read_csv('C:/Users/NMAIL/Desktop/Research/counterfactual/data/lendingclub/dataset5/2007-2011/data.csv').to_numpy()

X, y = data[:, : -1], data[:, -1]
y = y.astype(int)

# sm = SMOTENC(categorical_features = [4, 5, 6, 7])
# sm = SMOTENC(categorical_features = [4, 5, 6])
ros = RandomOverSampler()

# X_res, y_res = sm.fit_resample(X, y)
X_res, y_res = ros.fit_resample(X, y)

# with open('C:/Users/NMAIL/Desktop/Research/counterfactual/data/lendingclub_dataset/data_2020_oversampled.pickle', 'wb') as f:
# 	pickle.dump(X_res, f)
# 	pickle.dump(y_res, f)

# with open('C:/Users/NMAIL/Desktop/Research/counterfactual/data/sampled_data.pickle', 'rb') as f:
# 	X_res = pickle.load(f)
# 	y_res = pickle.load(f)

print('Resampled dataset shape %s' % Counter(y_res))

y_res = y_res.reshape(-1, 1)

new_data = np.concatenate((X_res, y_res), axis = 1)

new_data = pd.DataFrame(new_data, columns = ['emp_length', 'annual_inc', 'open_acc', 'cr_history', 'grade', 'home_ownership', 'purpose', 'addr_state', 'loan_status'])
# new_data = pd.DataFrame(new_data, columns = ['emp_length', 'annual_inc', 'open_acc', 'cr_history', 'grade', 'home_ownership', 'purpose', 'loan_status'])

# new_data.to_csv('C:/Users/NMAIL/Desktop/Research/counterfactual/data/lendingclub/dataset5/2007-2020/data1_oversampled.csv', index = False)
new_data.to_csv('C:/Users/NMAIL/Desktop/Research/counterfactual/data/lendingclub/dataset5/2007-2011/data_oversampled1.csv', index = False)