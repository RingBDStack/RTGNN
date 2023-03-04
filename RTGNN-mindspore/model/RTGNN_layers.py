import mindspore
import mindspore.nn as nn
from mindspore import Parameter, Tensor, load_checkpoint
import mindspore.ops as ops
from mindspore.common.initializer import initializer, XavierUniform


class IntraGNN(nn.Cell):

    def __init__(self, raw_features_dim, hidden_dim, slope, dropout):
        super(IntraGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.w_trans = Parameter(
            Tensor(shape=(raw_features_dim, raw_features_dim),
                   dtype=mindspore.float32,
                   init=XavierUniform()))
        self.w_gnn = Parameter(
            Tensor(shape=(raw_features_dim, hidden_dim),
                   dtype=mindspore.float32,
                   init=XavierUniform()))
        self.leaky_relu = nn.LeakyReLU(slope)
        self.dropout = nn.Dropout(dropout)

    def construct(self, features, weights, edge_feats, RL_thresholds,
                  batch_idx):
        # batch_weights = weights[Tensor(batch_idx)]
        batch_weights = weights[batch_idx]
        num_neighs = ((batch_weights > 0.001) * 1).astype(mindspore.float32)
        num_neighs = ops.ReduceSum()(num_neighs, -1).astype(mindspore.float32)
        # num_neighs = ops.ceil(num_neighs * Tensor([RL_thresholds]))
        num_neighs = ops.ceil(num_neighs * RL_thresholds)
        adj_mat_sampled, view_score = filter_neighbors(batch_weights,
                                                       num_neighs, edge_feats)
        M, N, D = adj_mat_sampled.shape
        adj_mat_sampled = adj_mat_sampled + ops.eye(N, N, mindspore.float32)
        adj_mat_sampled[adj_mat_sampled > 1] = 1
        view_features = features[batch_idx]
        view_features = ops.matmul(adj_mat_sampled, view_features)
        view_features = ops.matmul(view_features, self.w_gnn)
        view_features = self.leaky_relu(view_features)
        # view_features = self.dropout(view_features)
        return view_features, view_score


class InterGNN(nn.Cell):

    def __init__(self, num_views, num_regions, hidden_dim, dropout, slope,
                 inter_type, attn_vec_dim, mat2vec):
        super(InterGNN, self).__init__()
        self.inter_type = inter_type
        self.mat2vec = mat2vec
        # self.adj_gcn = nn.Parameter(torch.FloatTensor(num_regions, num_views, num_views))
        self.adj_gcn = Parameter(
            Tensor(shape=(num_views, num_views),
                   dtype=mindspore.float32,
                   init=XavierUniform()))
        # self.adj_gcn = nn.Parameter(torch.FloatTensor(1,num_views))
        self.w_gcn = Parameter(
            Tensor(shape=(hidden_dim, hidden_dim),
                   dtype=mindspore.float32,
                   init=XavierUniform()))
        # self.w_trans = initializer('xavier_uniform', self.w_trans.shape, mindspore.float32)
        # self.w_gnn = initializer('xavier_uniform', self.w_gnn.shape, mindspore.float32)
        self.w_gnn_concat = nn.Dense(hidden_dim * num_regions,
                                     attn_vec_dim,
                                     has_bias=True)
        self.w_gnn = nn.Dense(hidden_dim, attn_vec_dim, has_bias=True)
        self.vec_gnn = nn.Dense(attn_vec_dim, 1, has_bias=False)
        # nn.init.xavier_uniform_(self.w_gnn_concat.weight)
        # nn.init.xavier_uniform_(self.w_gnn.weight)
        # nn.init.xavier_uniform_(self.vec_gnn.weight)
        self.w_gnn_concat.weight = initializer('xavier_uniform',
                                               self.w_gnn_concat.weight.shape,
                                               mindspore.float32)
        self.w_gnn.weight = initializer('xavier_uniform',
                                        self.w_gnn.weight.shape,
                                        mindspore.float32)
        self.vec_gnn.weight = initializer('xavier_uniform',
                                          self.vec_gnn.weight.shape,
                                          mindspore.float32)
        self.tanh = ops.Tanh()
        self.softmax = nn.Softmax(axis=0)
        self.dropout = nn.Dropout(keep_prob=1 - dropout)
        self.leaky_relu = nn.LeakyReLU(slope)

    def construct(self, view_features_list):
        if self.inter_type == 'gcn':
            batch_features = ops.cat([
                ops.ExpandDims()(view_features, 0)
                for view_features in view_features_list
            ],
                                     axis=0).permute(1, 2, 0, 3)
            batch_features = ops.mean(ops.matmul(self.adj_gcn, batch_features),
                                      axis=2).squeeze(2)
            batch_features = transform_matrix_vectors(batch_features,
                                                      self.mat2vec)
        elif self.inter_type == 'gat':
            beta = []
            batch_features = []
            for view_features in view_features_list:
                view_features = transform_matrix_vectors(
                    view_features, self.mat2vec)
                batch_features.append(ops.ExpandDims()(view_features, 0))
                if self.mat2vec == 'concat':
                    f = self.tanh(self.w_gnn_concat(view_features))
                else:
                    f = self.tanh(self.w_gnn(view_features))
                f_mean = ops.mean(f, axis=0, keep_dims=True)
                b = self.vec_gnn(f_mean)
                beta.append(b)
            beta = ops.Concat(axis=0)(beta)
            beta = self.softmax(beta)
            beta = ops.ExpandDims()(beta, -1)
            beta = self.dropout(beta)
            batch_features = ops.Concat(axis=0)(batch_features)
            batch_features = self.leaky_relu(ops.ReduceSum()(
                beta * batch_features, 0))
        else:
            batch_features = ops.Concat(axis=0)([
                ops.ExpandDims()(view_features, 0)
                for view_features in view_features_list
            ])
            batch_features = ops.mean(batch_features, axis=0)
            batch_features = transform_matrix_vectors(batch_features,
                                                      self.mat2vec)
        return batch_features


def filter_neighbors(batch_weights, num_neighs, edge_feats):
    M, N, D = batch_weights.shape
    neighs = (batch_weights > 0.001) * 1
    neighs[neighs < 0.001] = 0
    adj_mat_sampled = ops.ZerosLike()(neighs)
    view_score = 0.0
    total_num_neighs = ops.ReduceSum()(num_neighs)
    for k in range(M):
        for j in range(N):
            num_samp = num_neighs[k][j]
            if num_samp > 0.0:
                neighs_idx = neighs[k][j].nonzero()
                norm_ops = ops.LpNorm(axis=1, p=2)
                # print(edge_feats[i][neighs_idx])
                if neighs_idx.shape == (0, 1) or neighs_idx.shape[0] == -1:
                    continue
                # elif neighs_idx.shape[0] == -1:
                #     neighs_feats = edge_feats[k][neighs_idx]
                #     center_feats = edge_feats[k][j]
                #     # distance = torch.cosine_similarity(neighs_feats,center_feats,dim=1)
                #     distance = ops.LpNorm(axis=0,
                #                           p=2)(neighs_feats - center_feats)
                #     # distance = distance/torch.max(distance)
                #     dis_import = ops.Ones(distance.shape) - distance
                #     # weight_import = torch.abs(batch_weights[i][j][neighs_idx])
                #     # con_socres = dis_import.mul(weight_import)
                #     con_import = dis_import
                #     adj_mat_sampled[k][j][neighs_idx] = 1.
                #     view_score += con_import
                else:
                    neighs_idx_shape = neighs_idx.shape[0]
                    neighs_idx = neighs_idx.squeeze()
                    neighs_feats = edge_feats[k][neighs_idx]
                    ops.Print()(neighs_feats)
                    # print(neighs_feats.shape[0])
                    center_feats = edge_feats[k][j]
                    center_feats = ops.ExpandDims()(center_feats, 0)
                    center_feats = mindspore.numpy.tile(
                        center_feats, (neighs_idx_shape, 1))
                    distance = norm_ops(neighs_feats - center_feats)
                    max_distance = distance.max()
                    distance = distance / max_distance
                    dis_import = ops.ones(distance.shape,
                                          mindspore.float32) - distance
                    con_import = dis_import
                    rank_socres, rank_indices = ops.Sort(
                        descending=False)(con_import)
                    rank_indices = [
                        int(neighs_idx[idx]) for idx in rank_indices
                    ]
                    if len(rank_indices) > num_samp:
                        adj_mat_sampled[k][j][
                            rank_indices[:int(num_samp)]] = 1.
                        view_score += ops.ReduceSum()(
                            rank_socres[:int(num_samp)])
                    else:
                        adj_mat_sampled[k][j][rank_indices] = 1.
                        view_score += ops.ReduceSum()(rank_socres)
    view_score = view_score / total_num_neighs
    return adj_mat_sampled, view_score


def transform_matrix_vectors(matrices, mat2vec):
    M, N, D = matrices.shape
    if mat2vec == 'concat':
        matrices = matrices.reshape(M, N * D, 1).squeeze()
    elif mat2vec == 'mean':
        matrices = ops.mean(matrices, axis=1)
    elif mat2vec == 'max':
        matrices, _ = ops.max(matrices, axis=1)
    return matrices


def RL_module(thresholds, RL_flags, rewords_log, view_scores_log, RL_setp):
    new_thresholds = thresholds.copy()
    new_RL_flags = RL_flags.copy
    new_rewords = [0] * len(RL_flags)
    for i in range(len(new_RL_flags)):
        if new_RL_flags[i] == True:
            if len(rewords_log[i]) >= 30 and abs(sum(
                    rewords_log[i][-10:])) <= 1:
                new_RL_flags[i] = False
                continue
            previous_epoch_scores = view_scores_log[-2][i]
            current_epoch_scores = view_scores_log[-1][i]
            reward = -1 if current_epoch_scores <= previous_epoch_scores else 1
            new_rewords[i] = reward
            new_thresholds[i] = thresholds[
                i] + RL_setp if reward == 1 else thresholds[i] - RL_setp
            new_thresholds[
                i] = 0.999 if new_thresholds[i] > 1 else new_thresholds[i]
            new_thresholds[
                i] = 0.001 if new_thresholds[i] < 0 else new_thresholds[i]
    return new_thresholds, new_RL_flags, new_rewords
