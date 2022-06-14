import os
import csv
import ast
import sys
import numpy as np
import pandas as pd

header = ['condition', 'num_profile', 'num_trial', 'profile', 'num_feat_fix', 'fixes', 'num_feat_ran', 'ranges', 'num_feat_pri', 'pris', 'num_feat_weight', 'weights', 'cf1', 'cf2', 'cf3']

def row_to_profile(row):

	profile = ''
	for info in row:
		profile += info + ','
	profile = profile[:-1]
	return profile

def reset_count():

	global finish_count
	finish_count = 0

def to_next_row():

	global row_idx, data, finish_count
	while data[row_idx] == '':
		row_idx += 1
		finish_count += 1

def preprocess(row):
	while row[-1] == ',':
		row = row[:-1]
		if len(row) == 0:
			break
	answer = []
	for char in row:
		if char == "'":
			answer.append('"')
		elif char == '"':
			answer.append("'")
		else:
			answer.append(char)

	return ''.join(answer)

def check_same(pf1, pf2):

	tmp1 = pf1.split(',')
	tmp2 = pf2.split(',')

	for i in range(9):
		if i in [0, 1, 2, 3, 8]:
			if tmp1[i] == '-' or tmp2[i] == '-':
				if tmp1[i] != tmp2[i]:
					return False
			elif float(tmp1[i]) != float(tmp2[i]):
				return False
		else:
			if tmp1[i] != tmp2[i]:
				return False
	return True


part_names = ['bongjae_choi', 'byunghyung_kim', 'byungsoo_kim', 'chaeeun_hwang', 'daeyoung_kim', 'geunwoo_kim', 'haram_kwon', 'heejeong_oh', 'hochang_lee', 'hyungjin_lee', 'hyunmuk_kang', 'jaehoon_choi', 'jaewon_kim', 'jeonghwan_kim', 'jiho_kwak', 'jinwoo_choi', 'jonghee_jeon', 'joowon_kim', 'juho_lee', 'jungwook_mun', 'junyong_park', 'min_kim', 'minjoon_kim', 'minwook_kim', 'nayoung_oh', 'sejoon_huh', 'seonggwang_kim', 'suhwan_song', 'wonjun_kang', 'yuji_roh']
# part_names = ['bongjae_choi']

action_idx = sys.argv[1]

if action_idx == '1':

	print("Start summing...", "\n")

	for name in part_names:
		print(name)

		result_path = 'C:/Users/NMAIL/Desktop/Research/counterfactual/exp_log/' + name + '_log_new.csv'

		if not os.path.exists(result_path):
			f = open(result_path, 'a', newline = '')
			writer = csv.writer(f)
			writer.writerow(header)
		else:
			f = open(result_path, 'a', newline = '')
			writer = csv.writer(f)

		source = 'C:/Users/NMAIL/Desktop/Research/counterfactual/experiment/' + name + '.csv'
		data = []
		with open(source, newline = '') as csvfile:
			reader = csv.reader(csvfile, quotechar = '|')
			for row in reader:
				data.append(','.join(row))
		csvfile.close()

		profile = None
		log_data = []
		row_idx = 0
		condition = 0
		num_profile = 0
		num_trial = 0
		finish_count = 0
		finish = False
		cfs = []
		fixes = []
		pris = []
		ranges = []
		weights = []


		while row_idx < len(data):

			cur_row = data[row_idx].split(',')[0]
			# print(row_idx)

			if 'new' in cur_row:
				# print('1')
				reset_count()
				new_condition = cur_row.split('.')[0][-1]
				if condition != new_condition:
					condition = new_condition
					num_profile = 1
					num_trial = 0
				else:
					num_profile += 1
					num_trial = 0
				row_idx += 1
				profile = data[row_idx]
				# log_data.append(condition)
				# log_data.append(num_profile)
				# log_data.append(num_trial)
				# log_data.append(profile)
				row_idx += 1

			elif 'cf' in cur_row:
				# print('2')
				reset_count()
				num_trial += 1
				log_data.append(condition)
				log_data.append(num_profile)
				log_data.append(num_trial)
				log_data.append(profile)
				row_idx += 1

				if condition != '1':
					while data[row_idx].split(',')[0] != 'feat_fix':
						cfs.append(data[row_idx])
						row_idx += 1

				else:
					while data[row_idx].split(',')[0] != '':
						cfs.append(data[row_idx])
						row_idx += 1
						if row_idx >= len(data):
							break
					for j in range(4):
						log_data.append(0)
						log_data.append([])
					for cf in cfs:
						log_data.append(cf)
					writer.writerow(log_data)
					fixes = []
					pris = []
					ranges = []
					weights = []

					log_data = []
					cfs = []

			elif 'feat_fix' in cur_row:
				# print('3')
				reset_count()
				row_idx += 1	
				if condition == '1' or condition == '3':
					log_data.append(0)
					log_data.append(fixes)
				else:
					cnt = 0
					feat_fix = data[row_idx].split(',')
					for feat in feat_fix:
						if feat != '':
							cnt += 1
							fixes.append(feat)
					log_data.append(cnt)
					log_data.append(fixes)
				row_idx += 1

			elif 'con_cat_ran' in cur_row:
				# print('4')
				reset_count()
				row_idx += 1
				if condition == '1' or condition == '3':
					log_data.append(0)
					log_data.append(ranges)
				else:
					cnt = 0
					# feat_ran = data[row_idx].split('","')
					feat_ran = preprocess(data[row_idx])
					if feat_ran == '':
						log_data.append(cnt)
						log_data.append(ranges)
					else:
						feat_ran = feat_ran.split("','")
						for feat in feat_ran:
							if feat[0] != "'":
								feat = "'" + feat
							if feat[-1] != "'":
								feat = feat + "'"
							tmp = eval(feat)
							if feat != '':
								cnt += 1
								ranges.append(tmp)
						log_data.append(cnt)
						log_data.append(ranges)
						
				row_idx += 1

			elif 'feat_pri' in cur_row:
				# print('5')
				reset_count()
				row_idx += 1
				if condition == '1' or condition == '2':
					log_data.append(0)
					log_data.append(pris)
				else:
					cnt = 0
					feat_pri = data[row_idx].split(',')
					for feat in feat_pri:
						if feat != '':
							cnt += 1
							pris.append(feat)
					log_data.append(cnt)
					log_data.append(pris)
				row_idx += 1

			elif 'weights' in cur_row:
				# print('6')
				reset_count()
				row_idx += 1
				if condition == '1' or condition == '2':
					log_data.append(0)
					log_data.append(weights)
				else:
					cnt = 0
					feat_weight = data[row_idx].split(',')
					for feat in feat_weight:
						if feat != '':
							weights.append(feat)
							if feat != '1':
								cnt += 1
					log_data.append(cnt)
					log_data.append(weights)
				row_idx += 1

				# print('before')
				# print(log_data)
				# print(cfs)
				for cf in cfs:
					log_data.append(cf)
				# print('here')

				writer.writerow(log_data)
				log_data = []
				cfs = []
				fixes = []
				pris = []
				ranges = []
				weights = []


			if row_idx >= len(data):
				break

			while data[row_idx].split(',')[0] == '':
				finish_count += 1
				row_idx += 1

				# print(row_idx)
				# print(data[row_idx])
				# print()

				if row_idx >= len(data):
					finish = True
					break

			if finish:
				break

		f.close()

elif action_idx == '2':

	for name in part_names:

		print(name)

		source = 'C:/Users/NMAIL/Desktop/Research/counterfactual/exp_log/' + name + '_log.csv'
		dest = 'C:/Users/NMAIL/Desktop/Research/counterfactual/exp_log/' + name + '_log_new.csv'

		orig_df = pd.read_csv(source)
		new_df = pd.read_csv(dest)

		for column in header:

			if column not in ['fixes', 'ranges', 'pris', 'weights']:

				orig_data = orig_df[column].to_numpy()
				new_data = new_df[column].to_numpy()

				if column in ['profile', 'cf1', 'cf2', 'cf3']:
					for i in range(len(orig_data)):

						if type(orig_data[i]) == str and type(new_data[i]) == str:
							if not check_same(orig_data[i], new_data[i]):
								print(column)
								print(orig_data[i])
								print(new_data[i])
						else:
							if not (np.isnan(orig_data[i]) and np.isnan(new_data[i])):
								print(column)
								print(orig_data[i])
								# print(np.isnan(orig_data[i]))
								print(new_data[i])
				else:
					for i in range(len(orig_data)):
						if orig_data[i] != new_data[i]:
							# print(column)
							# print(orig_data[i])
							# print(new_data[i])
							# print(type(orig_data[i]))
							# print(type(new_data[i]))
							if not (np.isnan(orig_data[i]) and np.isnan(new_data[i])):
								print(column)
								print(orig_data[i])
								# print(np.isnan(orig_data[i]))
								print(new_data[i])
							# print(np.isnan(new_data[i]))
				print()

		# print(orig_df['found'].to_numpy().tolist())
		# print(orig_df)
		# print(new_df)
		new_df['found'] = orig_df['found'].to_numpy().tolist()
		new_df['chosen'] = orig_df['chosen'].to_numpy().tolist()

		new_df.to_csv(dest, header = header + ['found', 'chosen'], index = False)