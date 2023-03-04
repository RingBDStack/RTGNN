import mindspore
import mindspore.nn as nn
from mindspore import Parameter, Tensor
import mindspore.ops as ops
from model.RTGNN_layers import *
from mindspore import save_checkpoint


class OneLayerRTGNN(nn.Cell):

    def __init__(self, dataset, features, weights, num_views, instance_classes,
                 node_classes, hidden_dim, dropout, slope, RL_step, RL_start,
                 threshold_start, lambeda, inter_type, attn_vec_dim, mat2vec):
        super(OneLayerRTGNN, self).__init__()
        self.dataset = dataset
        self.features = Parameter(Tensor(features, mindspore.float32),
                                  requires_grad=False)
        self.weights = Parameter(Tensor(weights, mindspore.float32),
                                 requires_grad=False)
        self.num_views = num_views
        self.num_regions = features[0, 0, :, :].shape[0]
        self.mat2vec = mat2vec
        self.RL_epoch_socres = Parameter(Tensor([0.0] * num_views,
                                                mindspore.float32),
                                         requires_grad=False)
        self.RL_thresholds = Parameter(Tensor([threshold_start] * num_views,
                                              mindspore.float32),
                                       requires_grad=False)
        # self.RL_flags = Parameter(Tensor([True] * num_views),
        #                           requires_grad=False)
        # self.RL_setp = RL_step
        # self.RL_start = RL_start
        # self.RL_view_scores_log = []
        # self.RL_rewords_log = Parameter(Tensor([[0] * RL_start] * num_views,
        #                                        mindspore.int32),
        #                                 requires_grad=False)
        # self.RL_thresholds_log = Parameter(Tensor([[1.0] * num_views] *
        #                                           RL_start, mindspore.float32),
        #                                    requires_grad=False)
        # self.RL_rewords_log = Tensor([[0] * RL_start] * num_views,
        #                              mindspore.int32)
        # self.RL_thresholds_log = Tensor([[1.0] * num_views] * RL_start,
        #                                 mindspore.float32)
        self.fnns = nn.CellList()
        for i in range(num_views):
            self.fnns.append(
                nn.Dense(self.weights.shape[-1], node_classes, has_bias=True))
        self.lambeda = lambeda
        self.intra_gnns = nn.CellList()
        for i in range(num_views):
            self.intra_gnns.append(
                IntraGNN(self.features.shape[-1], hidden_dim, slope, dropout))
        self.inter_gnn = InterGNN(num_views, self.num_regions, hidden_dim,
                                  dropout, slope, inter_type, attn_vec_dim,
                                  mat2vec)
        self.output_fnn = nn.Dense(hidden_dim, instance_classes)
        self.output_fnn_concat = nn.Dense(hidden_dim * self.num_regions,
                                          instance_classes)
        # nn.init.xavier_uniform_(self.output_fnn.weight)
        # nn.init.xavier_uniform_(self.output_fnn_concat.weight)
        self.output_fnn.weight = initializer('xavier_uniform',
                                             self.output_fnn.weight.shape,
                                             mindspore.float32)
        self.output_fnn_concat.weight = initializer(
            'xavier_uniform', self.output_fnn_concat.weight.shape,
            mindspore.float32)
        self.tanh = nn.Tanh()
        self.loss_function = nn.CrossEntropyLoss()

    def construct(self, batch_idx, train_flag, epoch, iter, num_batchs):
        edge_predicts = []
        for i in range(self.num_views):
            edge_predict = self.tanh(self.fnns[i](
                self.features[batch_idx][:, i, :, :]))  #Tensor
            edge_predicts.append(edge_predict)
        view_features_list = []
        view_scores_list = []
        for i, intra_gnn in enumerate(self.intra_gnns):
            # weights = self.weights[:, i, :, :]
            # edge_feats = edge_predicts[i]
            # RL_thresholds = self.RL_thresholds[i]
            # batch_idx = batch_idx
            # batch_weights = weights[batch_idx]
            # num_neighs = (batch_weights > 0.001) * 1
            # num_neighs = ops.ReduceSum()(num_neighs.astype(mindspore.float32),
            #                              -1)
            # num_neighs = ops.ceil(num_neighs * RL_thresholds)
            view_features, view_score = intra_gnn(self.features[:, i, :, :],
                                                  self.weights[:, i, :, :],
                                                  edge_predicts[i],
                                                  self.RL_thresholds[i],
                                                  batch_idx)
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
        return batch_features, gnn_predicts, edge_predicts
