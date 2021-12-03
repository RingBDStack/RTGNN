import torch
import torch.nn as nn
from model.RTGNN_DBLP_layers import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class OneLayerRTGNN_DBLP(nn.Module):
	def __init__(self,
				 features, weights,
				 num_views, instance_classes, node_classes, hidden_dim,
				 dropout, slope,
				 RL_step, RL_start, threshold_start,
				 lambeda,
				 inter_type, attn_vec_dim):
		super(OneLayerRTGNN_DBLP, self).__init__()
		self.features = nn.Parameter(features, requires_grad=False)
		self.weights = nn.Parameter(weights, requires_grad=False)
		self.num_views = num_views
		self.RL_epoch_socres = [0.0]*num_views
		self.RL_thresholds = [threshold_start]*num_views
		self.RL_flags = [True]*num_views
		self.RL_setp = RL_step
		self.RL_start = RL_start
		self.RL_view_scores_log = []
		self.RL_rewords_log = [[0]*RL_start]*num_views
		self.RL_thresholds_log = [[1.0]*num_views]*RL_start
		self.fnn = nn.Linear(features.size()[-1], node_classes, bias=True)
		self.lambeda = lambeda
		self.intra_gnns = nn.ModuleList()
		for i in range(num_views):
			self.intra_gnns.append(IntraGNN(features.size()[-1], hidden_dim, slope))
		self.inter_gnn = InterGNN(num_views, hidden_dim,
								  dropout, slope,
								  inter_type, attn_vec_dim)
		self.output_fnn = nn.Linear(hidden_dim, instance_classes)
		nn.init.xavier_uniform_(self.output_fnn.weight)
		self.tanh = nn.Tanh()
		self.loss_function = nn.CrossEntropyLoss()
	def forward(self, input):
		batch_idx, batch_labels, train_flag, epoch, iter, num_batchs = input
		view_features_list = []
		view_scores_list = []
		for i, intra_gnn in enumerate(self.intra_gnns):
			view_features, view_score = intra_gnn(self.features,
												  self.weights[i],
												  self.features,
												  self.RL_thresholds[i], batch_idx)
			view_features_list.append(view_features)
			view_scores_list.append(view_score)
		batch_features = self.inter_gnn(view_features_list)
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
		return batch_features, batch_labels, gnn_predicts, gnn_predicts, train_flag
	def loss(self, input):
		batch_features, batch_labels, gnn_predicts, edge_predicts, train_flag = self.forward(input)
		batch_labels = torch.LongTensor(batch_labels).to(device)
		loss_gnn = self.loss_function(gnn_predicts, batch_labels)
		return loss_gnn, batch_features

