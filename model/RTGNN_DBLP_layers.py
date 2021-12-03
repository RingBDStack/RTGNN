import torch
import torch.nn as nn
import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IntraGNN(nn.Module):
	def __init__(self,raw_features_dim, hidden_dim, slope):
		super(IntraGNN, self).__init__()
		self.hidden_dim = hidden_dim
		self.w_trans = nn.Parameter(torch.FloatTensor(raw_features_dim, raw_features_dim))
		self.w_gnn = nn.Parameter(torch.FloatTensor(raw_features_dim, hidden_dim))
		nn.init.xavier_uniform_(self.w_trans)
		nn.init.xavier_uniform_(self.w_gnn)
		self.leaky_relu = nn.LeakyReLU(slope)
	def forward(self, features, weights, edge_feats, RL_thresholds, batch_idx):
		batch_weights = weights[torch.LongTensor(batch_idx).to(device)]
		num_neighs = (batch_weights > 0) * 1
		num_neighs = torch.sum(num_neighs, dim=-1).float()
		num_neighs = torch.ceil(num_neighs * torch.FloatTensor([RL_thresholds]).to(device)).int()
		adj_mat_sampled, view_score = filter_neighbors(batch_weights, num_neighs, edge_feats)
		view_features = torch.matmul(adj_mat_sampled, features)
		view_features = torch.matmul(view_features, self.w_gnn)
		view_features = self.leaky_relu(view_features)
		return view_features, view_score

class InterGNN(nn.Module):
	def __init__(self,num_views, hidden_dim,
				 dropout, slope,
				 inter_type, attn_vec_dim):
		super(InterGNN, self).__init__()
		self.inter_type = inter_type
		# self.adj_gcn = nn.Parameter(torch.FloatTensor(1,num_views))
		self.adj_gcn = nn.Parameter(torch.FloatTensor(num_views, num_views))
		self.w_gcn = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
		nn.init.xavier_uniform_(self.adj_gcn)
		nn.init.xavier_uniform_(self.w_gcn)
		self.w_gnn = nn.Linear(hidden_dim, attn_vec_dim, bias=True)
		self.vec_gnn = nn.Linear(attn_vec_dim, 1, bias=False)
		nn.init.xavier_uniform_(self.w_gnn.weight)
		nn.init.xavier_uniform_(self.vec_gnn.weight)
		self.tanh = nn.Tanh()
		self.softmax = nn.Softmax(dim=0)
		self.dropout = nn.Dropout(dropout)
		self.leaky_relu = nn.LeakyReLU(slope)
	def forward(self, view_features_list):
		if self.inter_type == 'gcn':
			batch_features = torch.cat([view_features.unsqueeze(0) for view_features in view_features_list],dim=0).permute(1, 0, 2)
			batch_features = torch.mean(torch.matmul(self.adj_gcn, batch_features),dim=1).squeeze()
		elif self.inter_type == 'mean':
			batch_features = torch.cat([view_features.unsqueeze(0) for view_features in view_features_list], dim=0)
			batch_features = torch.mean(batch_features,dim=0)
		return batch_features

def filter_neighbors(batch_weights, num_neighs, edge_feats):
	M, N = batch_weights.shape
	neighs = (batch_weights > 0) * 1
	adj_mat_sampled = torch.zeros(neighs.shape).to(device)
	view_score = 0.0
	total_num_neighs = torch.sum(num_neighs).item()
	for i in range(M):
		num_samp = num_neighs[i].item()
		if num_samp > 0.0:
			neighs_idx = neighs[i].nonzero().squeeze()
			if neighs_idx.shape == torch.Size([0]):
				continue
			elif neighs_idx.shape == torch.Size([]):
				weight_import = batch_weights[i][neighs_idx]
				# con_socres = dis_import.mul(weight_import)
				con_import = weight_import
				adj_mat_sampled[i][neighs_idx.item()] = weight_import
				view_score += con_import.item()
			else:
				weight_import = batch_weights[i][neighs_idx]
				# con_socres = dis_import.mul(weight_import)
				con_import = weight_import
				rank_socres, rank_indices = torch.sort(con_import, descending=True)
				rank_indices = [int(neighs_idx[idx]) for idx in rank_indices]
				if len(rank_indices) > num_samp:
					adj_mat_sampled[i][rank_indices[:num_samp]] = rank_socres[:num_samp]
					view_score += torch.sum(rank_socres[:num_samp]).item()
				else:
					adj_mat_sampled[i][rank_indices] = rank_socres
					view_score += torch.sum(rank_socres).item()
		adj_mat_sampled[i] = adj_mat_sampled[i]/torch.max(adj_mat_sampled[i])
	view_score = view_score/total_num_neighs
	return adj_mat_sampled, view_score

def RL_module(thresholds, RL_flags, rewords_log, view_scores_log, RL_setp):
	new_thresholds = copy.deepcopy(thresholds)
	new_RL_flags = copy.deepcopy(RL_flags)
	new_rewords = [0]*len(RL_flags)
	for i in range(len(new_RL_flags)):
		if new_RL_flags[i] == True:
			if len(rewords_log[i])>=10 and abs(sum(rewords_log[i][-10:])) <= 1:
				new_RL_flags[i]=False
				continue
			previous_epoch_scores = view_scores_log[-2][i]
			current_epoch_scores = view_scores_log[-1][i]
			reward = -1 if current_epoch_scores <= previous_epoch_scores else 1
			new_rewords[i] = reward
			new_thresholds[i] = thresholds[i] + RL_setp if reward == 1 else thresholds[i] - RL_setp
			new_thresholds[i] = 0.999 if new_thresholds[i] > 1 else new_thresholds[i]
			new_thresholds[i] = 0.001 if new_thresholds[i] < 0 else new_thresholds[i]
	return new_thresholds, new_RL_flags, new_rewords

