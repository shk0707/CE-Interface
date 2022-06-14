import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys, getopt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from easydict import EasyDict as edict


class Compas(Dataset):

	def __init__(self, data):
		super(Compas, self).__init__()
		self.data = data.to_numpy()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):

		item = self.data[idx]
		input_data = torch.from_numpy(item[ : -1]).float()
		label_data = torch.tensor(int(item[-1])).type(torch.LongTensor)

		return input_data, label_data

class CompasNet(nn.Module):

	def __init__(self):
		super(CompasNet, self).__init__()

		self.linear_1 = nn.Linear(12, 20)
		self.relu_1 = nn.ReLU()
		# self.linear_2 = nn.Linear(128, 32)
		# self.relu_2 = nn.ReLU()
		# self.linear_3 = nn.Linear(32, 4)
		# self.relu_3 = nn.ReLU()
		# self.linear_4 = nn.Linear(4, 1)
		# self.relu_4 = nn.ReLU()
		self.output = nn.Linear(20, 1)
		# self.softmax = nn.Softmax(dim = 1)
		self.sig = nn.Sigmoid()

		# self.sig = nn.Sigmoid()

	def forward(self, x):
		x = self.linear_1(x)
		x = self.relu_1(x)
		# x = self.linear_2(x)
		# x = self.relu_2(x)
		# x = self.linear_3(x)
		# x = self.relu_3(x)
		# x = self.linear_4(x)
		# x = self.relu_4(x)
		# x = self.sig(x)
		x = self.output(x)
		# x = self.softmax(x)
		x = self.sig(x)
		return x


def preprocess(data_path, indices):

	data = pd.read_csv(data_path)
	cont_idx, cat_idx, label_idx = indices[0], indices[1], indices[2]
	cont_features = [data.columns[idx] for idx in cont_idx]
	cat_features = [data.columns[idx] for idx in cat_idx]
	
	cont_data = data[cont_features]
	cat_data = data[cat_features]
	label = data[data.columns[label_idx]]

	# tmp = pd.concat([cont_data, cat_data], axis = 1)
	# tmp = pd.concat([tmp, label], axis = 1)

	# Preprocess continuous data by scaling between 0 and 1
	minmax = MinMaxScaler()
	for feature in cont_data.columns:
		cont_data[feature] = minmax.fit_transform(np.array(cont_data[feature]).reshape(-1, 1))

	# Preprocess categorical data by one-hot encoding
	cat_data = pd.get_dummies(cat_data)

	data = pd.concat([cont_data, cat_data], axis = 1)
	data = pd.concat([data, label], axis = 1)

	# tmp.to_csv('C:/Users/NMAIL/Desktop/Looxid/codes/compas_data.csv')

	return data


def get_dataloader(data, batch_size):

	msk = np.random.rand(len(data)) < 0.8

	train_data = data[msk]
	test_data = data[msk]

	train_dataset = Compas(train_data)
	test_dataset = Compas(test_data)

	train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
	test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, drop_last = False)

	return train_dataloader, test_dataloader


if __name__ == "__main__":

	csv_data_path = 'C:/Users/NMAIL/Desktop/Looxid/compas_dataset/data.csv'
	indices = [[0, 3], [1, 2, 4], 8]

	args = edict()
	args.batch_size = 64
	args.epoch = 100000
	args.gpu = True
	args.lr = 0.0001

	# Preprocess data
	data = preprocess(csv_data_path, indices)

	# Train/Test data loader
	train_dataloader, test_dataloader = get_dataloader(data, args.batch_size)

	# Train/Test
	device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'

	net = CompasNet().to(device)

	optimizer = optim.Adam(net.parameters(), lr = args.lr)
	criterion = nn.L1Loss()

	best_loss = 100

	for epoch in range(args.epoch):

		net.train()
		train_loss = 0
		train_accuracy = 0
		train_data_num = 0
		for x, y in train_dataloader:

			x = x.to(device)
			y = y.to(device)

			pred_y = net(x).flatten()
			loss = criterion(pred_y, y)
			
			train_loss += loss.item() * x.shape[0]
			accuracy = (pred_y.round() == y).float().mean()
			train_accuracy += accuracy.item() * x.shape[0]
			train_data_num += x.shape[0]

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if (epoch + 1) % 100 == 0:
			print("Epoch: ", epoch + 1)
			print("Training loss: ", train_loss / train_data_num)
			print("Training accuracy: ", train_accuracy / train_data_num)
			print()

			net.eval()
			with torch.no_grad():
				test_loss = 0
				test_accuracy = 0
				test_data_num = 0

				for x, y in test_dataloader:

					x = x.to(device)
					y = y.to(device)

					pred_y = net(x).flatten()
					loss = criterion(pred_y, y)
					
					test_loss += loss.item() * x.shape[0]
					accuracy = (pred_y.round() == y).float().mean()
					test_accuracy += accuracy.item() * x.shape[0]
					test_data_num += x.shape[0]

				print("Test loss: ", test_loss / test_data_num)
				print("Test accuracy: ", test_accuracy / test_data_num)
				print()

				if test_loss / test_data_num < best_loss:
					torch.save(net, 'C:/Users/NMAIL/Desktop/Looxid/codes/compas_model.pth')
					best_loss = test_loss / test_data_num
					print('saved')
					print()