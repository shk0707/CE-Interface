import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import sys, getopt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from easydict import EasyDict as edict


class LClub(Dataset):

	def __init__(self, data):
		super(LClub, self).__init__()
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):

		item = self.data[idx]
		input_data = torch.from_numpy(item[ : -1]).float()
		label_data = torch.from_numpy(item[-1 : ]).float()

		return input_data, label_data.type(torch.LongTensor)

class LClubNet(nn.Module):

	def __init__(self):
		super(LClubNet, self).__init__()

		# self.linear_1 = nn.Linear(70, 5)
		# self.linear_1 = nn.Linear(69, 5)
		self.linear_1 = nn.Linear(69, 1)
		# self.linear_1 = nn.Linear(19, 5)
		# self.relu_1 = nn.ReLU()
		# self.output = nn.Linear(5, 1)
		# self.sig = nn.Sigmoid()

	def forward(self, x):
		x = self.linear_1(x)
		# x = self.relu_1(x)
		# x = self.output(x)
		# x = self.sig(x)
		return x


def preprocess(data_path):

	data = pd.read_csv(data_path)
	
	cont_features = ['emp_length', 'annual_inc', 'open_acc', 'cr_history']
	cat_features = ['grade', 'home_ownership', 'purpose', 'addr_state']
	# cat_features = ['grade', 'home_ownership', 'purpose']

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

	return data.to_numpy()


def get_dataloader(data, batch_size):

	msk = np.random.rand(len(data)) < 0.8

	train_data = data[msk]
	test_data = data[~msk]

	# train_0, train_1, test_0, test_1 = 0, 0, 0, 0

	# for data in train_data:
	# 	if data[-1] == 1:
	# 		train_1 += 1
	# 	else:
	# 		train_0 += 1

	# for data in test_data:
	# 	if data[-1] == 1:
	# 		test_1 += 1
	# 	else:
	# 		test_0 += 1

	# print(train_0, train_1, test_0, test_1)
	# exit()

	train_dataset = LClub(train_data)
	test_dataset = LClub(test_data)

	train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
	test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, drop_last = False)

	return train_dataloader, test_dataloader


def init_weights(m):
	if type(m) == nn.Linear:
		torch.nn.init.zeros_(m.weight)
		torch.nn.init.zeros_(m.bias)

if __name__ == "__main__":

	# csv_data_path = 'C:/Users/NMAIL/Desktop/Research/counterfactual/data/lendingclub/dataset5/2007-2020/data1_oversampled.csv'
	csv_data_path = 'C:/Users/NMAIL/Desktop/Research/counterfactual/data/lendingclub/dataset5/2007-2011/data_oversampled1.csv'

	if len(sys.argv) > 1:
		last_epoch = int(sys.argv[1])
		# last_ckpt_path = 'C:/Users/NMAIL/Desktop/Research/counterfactual/codes/ckpt16-20_noaddr/' + str(last_epoch) + '.pt'
		last_ckpt_path = 'C:/Users/NMAIL/Desktop/Research/counterfactual/codes/ckpt07-11_os1/' + str(last_epoch) + '.pt'
	else:
		last_epoch, last_ckpt_path = 0, None

	# Hyper-parameters
	batch_size = 128
	epoch = 10000
	gpu = True
	lr = 0.001

	# Preprocess data and convert to numpy
	data = preprocess(csv_data_path)

	# Train/Test data loader
	train_dataloader, test_dataloader = get_dataloader(data, batch_size)

	# Train/Test
	device = 'cuda' if torch.cuda.is_available() and gpu else 'cpu'

	# Model Initiation
	if last_ckpt_path != None:
		net = torch.load(last_ckpt_path)
	else:
		net = LClubNet().to(device)
		# net.apply(init_weights)


	# Optimizer and Loss function
	optimizer = optim.Adam(net.parameters(), lr = lr)
	# criterion = nn.L1Loss()
	criterion = nn.MSELoss()

	print()
	print("Training start")

	best_loss = 100
	best_acc = 0

	for epoch in range(epoch):

		if epoch < last_epoch:
			continue

		net.train()
		train_loss = 0
		train_acc = 0
		train_data_num = 0
		i = 0
		print("Epoch: ", epoch + 1)
		print()
		for x, y in train_dataloader:

			i += 1

			x = x.to(device)
			y = y.to(device).float()

			pred_y = net(x)

			loss = criterion(pred_y, y)

			train_loss += loss.item() * x.shape[0]
			accuracy = (pred_y.round() == y).float().mean()
			train_acc += accuracy.item() * x.shape[0]
			train_data_num += x.shape[0]

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if i % 3000 == 0:
				print("Mini-batch: ", i)
				print("Mini-batch loss: ", train_loss / train_data_num)
				print("Mini-batch accuracy: ", train_acc / train_data_num)
				print()

		if epoch % 20 == 1:
			# torch.save(net, 'C:/Users/NMAIL/Desktop/Research/counterfactual/codes/ckpt2020_os_1/' + str(epoch) + '.pt')
			# torch.save(net, 'C:/Users/NMAIL/Desktop/Research/counterfactual/codes/ckpt07-11_os1/' + str(epoch) + '.pt')
			torch.save(net, 'C:/Users/NMAIL/Desktop/Research/counterfactual/codes/tmp/' + str(epoch) + '.pt')

		print('Train loss: {:18.16f}, Train accuracy: {:18.16f}'.format(train_loss/train_data_num, train_acc/train_data_num))
		print()

		print("Evaluation start")
		net.eval()
		with torch.no_grad():
			test_loss = 0
			test_acc = 0
			test_data_num = 0
			test_00 = 0
			test_01 = 0
			test_10 = 0
			test_11 = 0
			for x, y in test_dataloader:

				x = x.to(device)
				y = y.to(device)

				pred_y = net(x)

				loss = criterion(pred_y, y)

				test_loss += loss.item() * x.shape[0]
				accuracy = (pred_y.round() == y).float().mean()
				test_acc += accuracy.item() * x.shape[0]
				test_data_num += x.shape[0]

				pred_y = pred_y.round()

				y0 = y == 0
				y1 = y == 1
				pred_y0 = pred_y == 0
				pred_y1 = pred_y == 1

				test_00 += torch.logical_and(y0, pred_y0).float().sum().item()
				test_01 += torch.logical_and(y0, pred_y1).float().sum().item()
				test_10 += torch.logical_and(y1, pred_y0).float().sum().item()
				test_11 += torch.logical_and(y1, pred_y1).float().sum().item()

			test_loss /= test_data_num
			test_acc /= test_data_num
			test_00 /= test_data_num
			test_01 /= test_data_num
			test_10 /= test_data_num
			test_11 /= test_data_num

			# print("Epoch: ", epoch + 1)
			print('Test loss:  {:18.16f}, Test accuracy:  {:18.16f}'.format(test_loss, test_acc))
			print('FF: {:18.16f}, FT: {:18.16f}, TF: {:18.16f}, TT: {:18.16f}'.format(test_00, test_01, test_10, test_11))

			if test_acc > best_acc and test_11 > 0.3 and test_00 > 0.3:
			# if test_acc > best_acc and test_11 > 0.05 and test_00 > 0.3:
				# torch.save(net, 'C:/Users/NMAIL/Desktop/Research/counterfactual/codes/ckpt2020_os_1/best.pt')
				# torch.save(net, 'C:/Users/NMAIL/Desktop/Research/counterfactual/codes/ckpt07-11_os1/best.pt')
				torch.save(net, 'C:/Users/NMAIL/Desktop/Research/counterfactual/codes/tmp/best.pt')
				best_loss = test_loss
				best_acc = test_acc
				print('New best accuracy - Saved!')
				print('Best accuracy: ', best_acc)
				print('Best 00: ', test_00)
				print('Best 01: ', test_01)
				print('Best 10: ', test_10)
				print('Best 11: ', test_11)
			print()