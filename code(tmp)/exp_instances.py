# import DiCE
import dice_ml
from dice_ml.utils import helpers # helper functions
import torch
import torch.nn as nn
import tensorflow as tf
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import random
import csv

# from lending_model import LClubNet

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class LClubNet(nn.Module):

	def __init__(self):
		super(LClubNet, self).__init__()

		self.linear_1 = nn.Linear(70, 5)
		# self.linear_1 = nn.Linear(69, 5)
		# self.linear_1 = nn.Linear(19, 5)
		self.relu_1 = nn.ReLU()
		self.output = nn.Linear(5, 1)
		self.sig = nn.Sigmoid()

	def forward(self, x):
		x = self.linear_1(x)
		x = self.relu_1(x)
		x = self.output(x)
		x = self.sig(x)
		return x

data = pd.read_csv('C:/Users/NMAIL/Desktop/Research/counterfactual/data/lendingclub/dataset5/2007-2011/data_oversampled.csv')
orig_data = pd.read_csv('C:/Users/NMAIL/Desktop/Research/counterfactual/data/lendingclub/dataset5/2007-2011/data.csv').to_numpy().tolist()

feat_vary = ['emp_length', 'annual_inc', 'open_acc', 'cr_history', 'grade', 'home_ownership', 'purpose', 'addr_state']
cont_ran = {}
cat_ran = {}
user_weight = {
	'emp_length':1, 
	'annual_inc':1, 
	'open_acc': 1, 
	'cr_history': 1, 
	'grade': 1, 
	'home_ownership':1, 
	'purpose':1, 
	'addr_state':1}

d = dice_ml.Data(dataframe=data, continuous_features=['emp_length', 'annual_inc', 'open_acc', 'cr_history'], outcome_name='loan_status', permitted_range = cont_ran)

cont_features = ['emp_length', 'annual_inc', 'open_acc', 'cr_history']
cat_features = ['grade', 'home_ownership', 'purpose', 'addr_state']

cont_data = data[cont_features]
cat_data = data[cat_features]
label = data['loan_status']

# Preprocess continuous data by scaling between 0 and 1
minmax = MinMaxScaler()
for feature in cont_data.columns:
	cont_data[feature] = minmax.fit_transform(np.array(cont_data[feature]).reshape(-1, 1))

# Preprocess categorical data by one-hot encoding
cat_data = pd.get_dummies(cat_data)

data = pd.concat([cont_data, cat_data], axis = 1)
data = pd.concat([data, label], axis = 1)
data = data.to_numpy()


backend = 'PYT'
ML_modelpath = 'C:/Users/NMAIL/Desktop/Research/counterfactual/codes/ckpt07-11_os/best.pt'
m = dice_ml.Model(model_path= ML_modelpath, backend=backend)

# initiate DiCE
exp = dice_ml.Dice(d, m)

net = torch.load(ML_modelpath)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


print("Start checking data")
print("Total ", len(data), " instances to check")


exp_ins = []
num_ins = 0

indice = list(range(len(data)))
random.shuffle(indice)


for i in range(len(orig_data)):

	ins = data[i]

	if ins[-1] == 0:

		x = torch.from_numpy(ins[:-1].astype('float64')).float().to(device)
		pred_y = net(x)

		if pred_y.round().item() == 0:

			query_instance = {'emp_length':orig_data[i][0], 'annual_inc':orig_data[i][1], 'open_acc':orig_data[i][2], 'cr_history':orig_data[i][3], 'grade':orig_data[i][4], 'home_ownership':orig_data[i][5], 'purpose':orig_data[i][6], 'addr_state':orig_data[i][7]}


			dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=4, desired_class="opposite", user_weight = user_weight, features_to_vary = feat_vary, permitted_option = cat_ran)
			message = dice_exp.visualize_as_list(show_only_changes=True)

			cfs = []
			for result in message:
				if float(result[-1]) > 0.5 and result[1] != 0 and result[2] != 0 and result not in cfs:
					cfs.append(result)

			if len(cfs) > 0:
				orig_data[i].append(len(cfs))
				with open('C:/Users/NMAIL/Desktop/Research/counterfactual/data/lendingclub/dataset5/2007-2011/expins_1.csv', 'a', newline = '') as f:
					writer = csv.writer(f)
					writer.writerow(orig_data[i])
				num_ins += 1

				if num_ins % 10 == 0 and num_ins >= 10:
					print(num_ins, " query instances found")

	if num_ins >= 100:
		break

	if i % 100 == 0:
		print(i, " instances checked")


# exp_ins = np.array(exp_ins)

# exp_ins = pd.DataFrame(exp_ins, columns = ['emp_length', 'annual_inc', 'open_acc', 'cr_history', 'grade', 'home_ownership', 'purpose', 'addr_state', 'loan_status', 'num_cf'])
# exp_ins = pd.DataFrame(exp_ins, columns = ['emp_length', 'annual_inc', 'open_acc', 'cr_history', 'grade', 'home_ownership', 'purpose', 'addr_state', 'loan_status'])

# with open('C:/Users/NMAIL/Desktop/Research/counterfactual/data/lendingclub_dataset/expins_2020.pickle', 'wb') as f:
# 	pickle.dump(exp_ins, f)

# exp_ins.to_csv('C:/Users/NMAIL/Desktop/Research/counterfactual/data/lendingclub/dataset5/2007-2020/expins_500.csv', index = False)
