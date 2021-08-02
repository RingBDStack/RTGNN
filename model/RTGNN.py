import torch
import torch.nn as nn
from model.RTGNN_layers import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class OneLayerRTGNN(nn.Module):
	def __init__(self,
				 features, weights,
				 num_views, instance_classes, node_classes, hidden_dim,
				 dropout, slope,
				 RL_step, RL_start, threshold_start,
				 lambeda,
				 inter_type, attn_vec_dim, mat2vec):
		super(OneLayerRTGNN, self).__init__()
		self.features = nn.Parameter(features, requires_grad=False)
		self.weights = nn.Parameter(weights, requires_grad=False)
		self.num_views = num_views
		self.num_regions = features[0,0,:,:].size()[0]
		self.mat2vec = mat2vec
		self.RL_epoch_socres = [0.0]*num_views
		self.RL_thresholds = [threshold_start]*num_views
		self.RL_flags = [True]*num_views
		self.RL_setp = RL_step
		self.RL_start = RL_start
		self.RL_view_scores_log = []
		self.RL_rewords_log = [[0]*RL_start]*num_views
		self.RL_thresholds_log = [[1.0]*num_views]*RL_start
		self.fnns = nn.ModuleList()
		for i in range(num_views):
			self.fnns.append(nn.Linear(self.weights.shape[-1], node_classes, bias=True))
		self.lambeda = lambeda
		self.intra_gnns = nn.ModuleList()
		for i in range(num_views):
			self.intra_gnns.append(IntraGNN(self.features.shape[-1], hidden_dim, slope,dropout))
		self.inter_gnn = InterGNN(num_views, self.num_regions, hidden_dim,
								  dropout, slope,
								  inter_type, attn_vec_dim, mat2vec)
		self.output_fnn = nn.Linear(hidden_dim, instance_classes)
		self.output_fnn_concat = nn.Linear(hidden_dim*self.num_regions, instance_classes)
		nn.init.xavier_uniform_(self.output_fnn.weight)
		nn.init.xavier_uniform_(self.output_fnn_concat.weight)
		self.tanh = nn.Tanh()
		self.loss_function = nn.CrossEntropyLoss()

	def forward(self, input):
		batch_idx, batch_labels, regions_labels, train_flag, epoch, iter, num_batchs = input
		edge_predicts = []
		for i in range(self.num_views):
			edge_predict = self.tanh(self.fnns[i](self.features[batch_idx][:,i,:,:]))
			edge_predicts.append(edge_predict)
		view_features_list = []
		view_scores_list = []
		for i, intra_gnn in enumerate(self.intra_gnns):
			view_features, view_score = intra_gnn(self.features[:,i,:,:],
												  self.weights[:,i,:,:],
												  edge_predicts[i].clone().detach(),
												  self.RL_thresholds[i], batch_idx)
			view_features_list.append(view_features)
			view_scores_list.append(view_score)
		batch_features = self.inter_gnn(view_features_list)
		if self.mat2vec == 'concat':
			gnn_predicts = self.output_fnn_concat(batch_features)
		else:
			gnn_predicts = self.output_fnn(batch_features)
		if train_flag:
			for i, score in enumerate(view_scores_list):
				self.RL_epoch_socres[i] += score
			if iter == num_batchs-1:
				self.RL_view_scores_log.append(self.RL_epoch_socres)
				self.RL_epoch_socres = [0.0]*self.num_views
				if True in self.RL_flags and epoch >= self.RL_start-1:
					thresholds, RL_flags, rewords = RL_module(self.RL_thresholds,
														      self.RL_flags,
														      self.RL_rewords_log,
														      self.RL_view_scores_log,
														      self.RL_setp)
					self.RL_thresholds = thresholds
					self.RL_thresholds_log.append(self.RL_thresholds)
					self.RL_flags = RL_flags
					self.RL_rewords_log = [self.RL_rewords_log[i]+[rewords[i]] for i in range(self.num_views)]
					print('thresholds{}'.format(thresholds))
					print('RL_flags{}'.format(RL_flags))
					print('rewords{}'.format(rewords))
					print('RL_rewords_log{}'.format(self.RL_rewords_log))
		return batch_features, batch_labels, regions_labels, gnn_predicts, edge_predicts, train_flag

	def loss(self, input):
		batch_features, batch_labels, regions_labels, gnn_predicts, edge_predicts, train_flag = self.forward(input)
		batch_labels = torch.LongTensor(batch_labels).to(device)
		loss_gnn = self.loss_function(gnn_predicts, batch_labels)
		loss_edge = torch.tensor(0.).to(device)
		regions_labels = torch.LongTensor(regions_labels).to(device)
		if train_flag:
			for i in range(len(edge_predicts)):
				for j in range(len(edge_predicts[i])):
					loss_edge += self.loss_function(edge_predicts[i][j], regions_labels)
		return [loss_gnn,self.lambeda*loss_edge], batch_features

