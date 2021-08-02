import scipy.io as sio
from random import shuffle
from scipy import *
import tensorly as tl
import numpy as np
import scipy.sparse
import pandas as pd
from sklearn.cluster import KMeans
from collections import defaultdict
import calendar

def to_adjlist_DBLP(filename):
	file = open(filename, 'r')
	lines = file.readlines()[3:]
	file.close()
	adjlist = []
	weilist = np.zeros((len(lines),len(lines)))
	for i, line in enumerate(lines):
		tmp_list = [int(idx) for idx in line.strip().split(' ')]
		adjlist.append(tmp_list)
		for w in tmp_list:
			weilist[i][w] +=1
		weilist[i] = weilist[i]/np.max(weilist[i])
	return adjlist, weilist

def load_data(dataset='HIV'):
	prefix = './data/preprocessed/' + dataset
	if dataset != 'DBLP':
		instances_labels = np.load(prefix + '/' + dataset + '_ins_labels.npy')
		feats_tensor = np.load(prefix + '/' + dataset + '_feat_tensor.npy')
		weighted_adjs_tensor = np.load(prefix + '/' + dataset + '_adj_tensor.npy')
		regions_labels = np.load(prefix + '/' + dataset + '_reg_labels.npy')
		train_val_test_idx = np.load(prefix + '/' + dataset + '_train_val_test.npz')
		return feats_tensor, weighted_adjs_tensor, regions_labels, instances_labels, train_val_test_idx
	else:
		adjlist00, weilist00 = to_adjlist_DBLP(prefix + '/0/0-1-0.adjlist')
		adjlist01, weilist01 = to_adjlist_DBLP(prefix + '/0/0-1-2-1-0.adjlist')
		adjlist02, weilist02 = to_adjlist_DBLP(prefix + '/0/0-1-3-1-0.adjlist')
		features = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
		labels = np.load(prefix + '/labels.npy')
		train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
		return [adjlist00, adjlist01, adjlist02], \
			   [weilist00, weilist01, weilist02], \
			   features, \
			   labels, \
			   train_val_test_idx

class index_generator:
	def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
		if num_data is not None:
			self.num_data = num_data
			self.indices = np.arange(num_data)
		if indices is not None:
			self.num_data = len(indices)
			self.indices = np.copy(indices)
		self.batch_size = batch_size
		self.iter_counter = 0
		self.shuffle = shuffle
		if shuffle:
			np.random.shuffle(self.indices)
	def next(self):
		if self.num_iterations_left() <= 0:
			self.reset()
		self.iter_counter += 1
		return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size: self.iter_counter * self.batch_size])
	def num_iterations(self):
		return int(np.ceil(self.num_data / self.batch_size))
	def num_iterations_left(self):
		return self.num_iterations() - self.iter_counter
	def reset(self):
		if self.shuffle:
			np.random.shuffle(self.indices)
		self.iter_counter = 0

def HOSVD_00(X):
	M, V, N, D = X.shape
	A_list = []
	for i in range(2, 4):
		mat = tl.unfold(X, i)
		u, _, _ = np.linalg.svd(mat)
		A_list.append(u)
	G = np.empty(X.shape)
	for i in range(M):
		for j in range(V):
			x = X[i, j, :, :]
			x = np.matmul(A_list[0].transpose(), x)
			x = np.matmul(x, A_list[1])
			G[i, j, :, :] = x
	return G, A_list

def HOSVD_01(X):
	M, V, N, D = X.shape
	sum_X = 0
	for i in range(M):
		for j in range(V):
			x = X[i, j, :, :]
			sum_X = sum_X + np.matmul(x, x.transpose())
	u, _, v = np.linalg.svd(sum_X)
	G = np.empty(X.shape)
	for i in range(M):
		for j in range(V):
			x = X[i, j, :, :]
			x = np.matmul(u.transpose(), x)
			G[i, j, :, :] = x
	return G, [u, v]

def load_raw_data_brain(dataset):
	raw_data = sio.loadmat('../data/raw/' + dataset + '/' + dataset +'.mat')
	reg_labels = raw_data['reg_labels'].squeeze()
	np.save('../data/preprocessed/' + dataset + '/' + dataset + '_reg_labels.npy', reg_labels)
	ins_labels = raw_data['label'].squeeze()
	ins_labels[ins_labels<0] = 0
	np.save('../data/preprocessed/' + dataset + '/' + dataset + '_ins_labels.npy', ins_labels)
	# !!!note: the fmri and dti views of the initial data have been exchanged
	raw_data['fmri'] = np.expand_dims(raw_data['fmri'].transpose(2, 0, 1), 1)
	raw_data['dti'] = np.expand_dims(raw_data['dti'].transpose(2, 0, 1), 1)
	adj_feat_tensor = np.concatenate((raw_data['fmri'], raw_data['dti']), axis=1)
	np.save('../data/preprocessed/' + dataset + '/' + dataset + '_adj_tensor.npy', adj_feat_tensor)
	np.save('../data/preprocessed/' + dataset + '/' + dataset + '_feat_tensor.npy', adj_feat_tensor)

def load_raw_data_DC(years_list, quarters_list, months_list):
	raw_stations_list = []
	for year in years_list[:3]:
		for quarter in quarters_list:
			filename = '../data/raw/BikeDC/' + year + quarter +'.csv'
			csv_file = pd.read_csv(filename)
			start_station = set(csv_file['Start station'])
			end_station = set(csv_file['End station'])
			raw_stations_list.append(start_station&end_station)
			print(filename)
	for year in years_list[3:]:
		for month in months_list:
			filename = '../data/raw/BikeDC/' + year + month +'.csv'
			csv_file = pd.read_csv(filename)
			if year == '2020/2020' and int(month) >= 4:
				start_station = set(csv_file['start_station_name'])
				end_station = set(csv_file['end_station_name'])
			else:
				start_station = set(csv_file['Start station'])
				end_station = set(csv_file['End station'])
			raw_stations_list.append(start_station & end_station)
			print(filename)
	raw_stations = raw_stations_list[0].intersection(*raw_stations_list)
	raw_stations_name2id = {name:id for id,name in zip(range(len(raw_stations)),raw_stations)}
	print(len(raw_stations_name2id))
	print(raw_stations_name2id)
	np.save('../data/preprocessed/BikeDC/raw_stations_name2id.npy',raw_stations_name2id)

def split_train_val_test(dataset, ratio_list):
	if dataset == 'HIV' or dataset == 'BP':
		raw_data = sio.loadmat('../data/raw/' + dataset + '/' + dataset + '.mat')
		raw_data = raw_data['label'].squeeze()
		all_idx = [idx for idx in range(len(raw_data))]
		raw_data[raw_data < 0] = 0
	else:
		raw_data = np.array([q for y in range(6) for q in range(4) for l in range(3)])
		all_idx = [idx for idx in range(72)]
		np.save('../data/preprocessed/BikeDC/BikeDC_ins_labels.npy', raw_data)
	shuffle(all_idx)
	train_idx = all_idx[:ratio_list[0]]
	val_idx = all_idx[ratio_list[0]:ratio_list[0]+ratio_list[1]]
	test_idx = all_idx[ratio_list[0]+ratio_list[1]:]
	np.savez('../data/preprocessed/' + dataset + '/' + dataset + '_train_val_test.npz',train = train_idx, val=val_idx, test = test_idx)
	print(raw_data[train_idx])
	print(raw_data[val_idx])
	print(raw_data[test_idx])
	print(ratio_list)
	print(np.sum(raw_data[train_idx]))
	print(np.sum(raw_data[val_idx]))
	print(np.sum(raw_data[test_idx]))

def data_preprocess(dataset='HIV'):
	if dataset == 'HIV' or dataset == 'BP':
		load_raw_data_brain(dataset)
		ratio_list = [24,6,40] if dataset == 'HIV' else [33, 8, 56]
		split_train_val_test(dataset,ratio_list)
	elif dataset == 'BikeDC':
		years_list = [str(year) + '/' + str(year) for year in range(2015, 2021)]
		quarters_list = ['Q' + str(quarter) for quarter in range(1, 5)]
		months_list = ['0' + str(month) if month < 10 else str(month) for month in range(1, 13)]
		view_list = ['weekday','weekend','month']
		ratio_list = [24,6,42]
		# load_raw_data_DC(years_list, quarters_list, months_list)
		# get_loc_feats_DC()
		# # optional
		# kmeans_distances_regions_DC()
		# res_regions_DC()
		generate_adj_tensor_00_DC(view_list, years_list, quarters_list, months_list)
		# label_regions_DC()
		split_train_val_test(dataset, ratio_list)

if __name__ == '__main__':
	print('data processing...')
	# data_preprocess('HIV')
	# data_preprocess('BP')
	# data_preprocess('BikeDC')
	# The DBLP has been processed