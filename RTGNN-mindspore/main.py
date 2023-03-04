import argparse
import numpy as np
from toolbox.data_preprocess import *
from toolbox.early_stopping import *
from toolbox.evaluation import *
from model.RTGNN import OneLayerRTGNN
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from tqdm import tqdm
import time
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import save_checkpoint, Tensor, load_checkpoint, set_seed

ms.set_context(device_target='GPU', device_id=0)
ap = argparse.ArgumentParser(description='RTGNN')
ap.add_argument('--dataset', default='HIV', help='Dataset name')
ap.add_argument('--num-views', type=int, default=2, help='Number of views.')
ap.add_argument('--instance-classes',
                type=int,
                default=2,
                help='Number of instance types.')
ap.add_argument('--node-classes',
                type=int,
                default=2,
                help='Number of node types.')
ap.add_argument('--hidden-dim',
                type=int,
                default=64,
                help='Dimension of the feature matrix.')
ap.add_argument('--inter-type', default='gat', help='Types of inter gnns.')
ap.add_argument('--attn-vec-dim',
                type=int,
                default=128,
                help='Dimension of the attention vector.')
ap.add_argument('--mat2vec', default='mean', help='Vectorization.')
ap.add_argument('--dropout', type=float, default=0.5, help='Dropout.')
ap.add_argument('--slope',
                type=float,
                default=0.2,
                help='The slope of Leaky Relu')
# ap.add_argument('--lr', type=float, default=0.005, help='Learning rate.')
ap.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
ap.add_argument('--weight_decay',
                type=float,
                default=0.001,
                help='The weight decay of the optimizer.')
ap.add_argument('--lambeda', type=float, default=1.0, help='Edge loss weight.')
ap.add_argument('--RL-step',
                type=float,
                default=0.02,
                help='Action step size of reinforcement learning.')
ap.add_argument('--RL-start',
                type=int,
                default=2,
                help='The epoch at which reinforcement learning begins.')
ap.add_argument('--threshold-start',
                default=0.5,
                help='The initial thresholds.')
ap.add_argument('--repeat',
                type=int,
                default=20,
                help='Repeat the training and testing for N times.')
ap.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
ap.add_argument('--batch-size', type=int, default=4, help='Batch size.')
ap.add_argument('--patience', type=int, default=5, help='Patience.')
ap.add_argument('--save-postfix',
                default='RTGNN_HIV',
                help='Postfix for the saved model and result.')
args = ap.parse_args()


def plt_hot_map(adj_tensor):
    colormap = LinearSegmentedColormap.from_list("", [
        'floralwhite', 'darkorange', 'tomato', 'orangered', 'red', 'firebrick',
        'darkred'
    ])
    plt.figure(figsize=(7, 6))
    h = sns.heatmap(data=adj_tensor,
                    cmap=colormap,
                    annot=False,
                    cbar=False,
                    vmax=1.0,
                    vmin=0.0,
                    xticklabels=20,
                    yticklabels=20)
    cb = h.figure.colorbar(h.collections[0])
    cb.ax.tick_params(labelsize=32)
    h.set_ylim([len(adj_tensor), 0])
    h.spines['top'].set_visible(True)
    h.spines['right'].set_visible(True)
    h.spines['bottom'].set_visible(True)
    h.spines['left'].set_visible(True)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlabel('Node', fontsize=34)
    plt.ylabel('Node', fontsize=34)
    plt.tight_layout()
    plt.show()


def RL_module(thresholds, RL_flags, rewords_log, view_scores_log, RL_setp):
    new_thresholds = thresholds.copy()
    new_RL_flags = RL_flags.copy()
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


class Loss(nn.LossBase):

    def __init__(self):
        super(Loss, self).__init__()

    def construct(self, batch_features, batch_labels, regions_labels,
                  gnn_predicts, edge_predicts, train_flag):
        loss_function = nn.SoftmaxCrossEntropyWithLogits(reduction='mean',
                                                         sparse=True)
        loss_gnn = loss_function(gnn_predicts, batch_labels)
        loss_edge = Tensor(0., mindspore.float32)
        if train_flag:
            for i in range(len(edge_predicts)):
                for j in range(edge_predicts[i].shape[0]):
                    loss_edge += loss_function(edge_predicts[i][j],
                                               regions_labels)
        return loss_gnn + loss_edge
        # return loss_gnn


def generate_reliable_adj_tensor(dataset, init_adjs_tensor):
    if dataset == 'HIV' or dataset == 'BP':
        _, U = HOSVD_01(init_adjs_tensor)
        U = U[0]
        M, V, N, N = init_adjs_tensor.shape
        reli_adj_tensor = np.zeros((M, V, N, N))
        init_feat_tensor = np.zeros((M, V, N, N))
        for m in tqdm(range(M)):
            for v in range(V):
                imp_matrix = np.abs(init_adjs_tensor[m, v, :, :])
                adj_matrix = np.zeros((N, N))
                tmp_matrix = np.zeros((N, N))
                feat_matrix = np.zeros((N, N))
                for i in range(N):
                    tmp_matrix[i][np.argpartition(
                        imp_matrix[i], -int(N / 2))[-int(N / 2):]] = 1
                    tmp_matrix[i][tmp_matrix[i] != 1] = 0
                tmp_matrix = tmp_matrix + np.transpose(tmp_matrix) + np.eye(N)
                tmp_matrix[tmp_matrix > 1] = 1
                for i in range(N):
                    for j in range(N):
                        if tmp_matrix[i][j] == 1:
                            adj_matrix[i][j] = np.exp(
                                -(1 / 2) * np.linalg.norm(U[j] - U[i], ord=2))
                        feat_matrix[i][j] = adj_matrix[i][j]
                    adj_matrix[i] = adj_matrix[i] / np.max(adj_matrix[i])
                    feat_matrix[i] = feat_matrix[i] / np.max(feat_matrix[i])
                    reli_adj_tensor[m, v, i] = adj_matrix[i]
                    init_feat_tensor[m, v, i] = feat_matrix[i]
    return init_feat_tensor, reli_adj_tensor


def RTGNN():
    print('RTGNN_{}...'.format(args.dataset))
    feats_tensor, weighted_adjs_tensor, regions_labels, instances_labels, train_val_test_idx = load_data(
        args.dataset)

    if args.dataset == 'HIV' or args.dataset == 'BP':
        feats_tensor, weighted_adjs_tensor = generate_reliable_adj_tensor(
            args.dataset, weighted_adjs_tensor)
    elif args.dataset == 'PROTEINS':
        feats_tensor, weighted_adjs_tensor = generate_reliable_adj_tensor(
            args.dataset, (weighted_adjs_tensor, feats_tensor))
    G, _ = HOSVD_01(feats_tensor)
    # G += feats_tensor
    features = Tensor(G, ms.float32)
    weighted_adjs = Tensor(weighted_adjs_tensor, ms.float32)
    train_idx = train_val_test_idx['train']
    val_idx = train_val_test_idx['val']
    test_idx = train_val_test_idx['test']
    svm_macro_f1_lists = []
    svm_micro_f1_lists = []
    kmeans_nmi_lists = []
    kmeans_ari_lists = []
    for _ in range(args.repeat):
        RL_start = args.RL_start
        RL_flags = [True] * args.num_views
        RL_step = args.RL_step
        RL_view_scores_log = []
        RL_rewords_log = [[0] * RL_start] * args.num_views
        RL_thresholds_log = [[1.0] * args.num_views] * RL_start
        model = OneLayerRTGNN(args.dataset, features, weighted_adjs,
                              args.num_views, args.instance_classes,
                              args.node_classes, args.hidden_dim, args.dropout,
                              args.slope, args.RL_step, args.RL_start,
                              args.threshold_start, args.lambeda,
                              args.inter_type, args.attn_vec_dim, args.mat2vec)
        optimizer = nn.Adam(model.trainable_params(),
                            learning_rate=args.lr,
                            weight_decay=args.weight_decay)
        model_stop = early_stopping(
            patience=args.patience,
            verbose=True,
            save_path='checkpoint/checkpoint_{}.ckpt'.format(
                args.save_postfix))
        loss_fn = Loss()
        train_idx_generator = index_generator(batch_size=args.batch_size,
                                              indices=train_idx)
        val_idx_generator = index_generator(batch_size=args.batch_size,
                                            indices=val_idx,
                                            shuffle=False)

        def forward_fn(batch_idx, batch_labels, regions_labels, train_flag,
                       epoch, iter, num_batchs):
            batch_features, gnn_predicts, edge_predicts = model(
                batch_idx, train_flag, epoch, iter, num_batchs)
            loss = loss_fn(batch_features, batch_labels, regions_labels,
                           gnn_predicts, edge_predicts, train_flag)
            return loss, batch_features

        grad_fn = ops.value_and_grad(forward_fn,
                                     None,
                                     optimizer.parameters,
                                     has_aux=True)

        def train_step(batch_idx, batch_labels, regions_labels, train_flag,
                       epoch, iter, num_batchs):
            (loss, _), grad = grad_fn(batch_idx, batch_labels, regions_labels,
                                      train_flag, epoch, iter, num_batchs)
            loss = ops.depend(loss, optimizer(grad))
            return loss, _

        for epoch in range(args.epochs):
            model.set_train()
            train_num_batchs = train_idx_generator.num_iterations()
            for iter in range(train_num_batchs):
                # print(iter)
                batch_train_idx = train_idx_generator.next()
                batch_train_labels = instances_labels[batch_train_idx]
                train_loss, _ = train_step(
                    Tensor(batch_train_idx, mindspore.int32),
                    Tensor(batch_train_labels, mindspore.int32),
                    Tensor(regions_labels, mindspore.int32), True, epoch, iter,
                    train_num_batchs)

                if iter == train_num_batchs - 1:
                    RL_view_scores_log.append(model.RL_epoch_socres)
                    # model.RL_epoch_socres = [0.0] * model.num_views
                    ops.assign(
                        model.RL_epoch_socres,
                        Tensor([0.0] * model.num_views, mindspore.float32))
                    if True in RL_flags and epoch >= RL_start - 1:
                        thresholds, RL_flags, rewords = RL_module(
                            model.RL_thresholds, RL_flags, RL_rewords_log,
                            RL_view_scores_log, RL_step)
                        ops.assign(model.RL_thresholds, thresholds)
                        RL_thresholds_log.append(model.RL_thresholds)
                        RL_rewords_log = [
                            RL_rewords_log[i] + [rewords[i]]
                            for i in range(args.num_views)
                        ]
                        print('thresholds{}'.format(thresholds))
                        print('RL_flags{}'.format(RL_flags))
                        print('rewords{}'.format(rewords))
                        print('RL_rewords_log{}'.format(RL_rewords_log))
                if iter % train_num_batchs == 0:
                    print(
                        f'Epoch {epoch:05d} | Iteration {iter:05d} | Train_Loss {train_loss}'
                    )
            #eval
            model.set_train(False)
            val_num_batchs = val_idx_generator.num_iterations()
            for _ in range(val_num_batchs):
                batch_val_idx = val_idx_generator.next()
                batch_val_labels = instances_labels[batch_val_idx]
                batch_feat, gnn_pred, edge_pred = model(
                    Tensor(batch_val_idx, mindspore.int32), False, _, _, _)
                val_loss = loss_fn(batch_feat,
                                   Tensor(batch_val_labels, mindspore.int32),
                                   Tensor(regions_labels, mindspore.int32),
                                   gnn_pred, edge_pred, False)
            print('Epoch {:05d} | Val_Loss {} '.format(epoch, val_loss))
            model_stop(val_loss, model)
            if model_stop.early_stop:
                print('Early stopping!')
                break
        model.set_train(False)
        ms.load_checkpoint(f'checkpoint/checkpoint_{args.save_postfix}.ckpt',
                           net=model)
        test_idx_generator = index_generator(batch_size=args.batch_size,
                                             indices=test_idx,
                                             shuffle=False)
        test_features_list = []
        test_num_batchs = test_idx_generator.num_iterations()
        for _ in range(test_num_batchs):
            batch_test_idx = test_idx_generator.next()
            batch_test_labels = instances_labels[batch_test_idx]
            batch_feat, gnn_pred, edge_pred = model(
                Tensor(batch_test_idx, mindspore.int32), False, _, _, _)
            val_loss = loss_fn(batch_feat,
                               Tensor(batch_test_labels, mindspore.int32),
                               Tensor(regions_labels, mindspore.int32),
                               gnn_pred, edge_pred, False)
            test_features_list.append(batch_feat)
        test_features = ms.ops.Concat()(test_features_list)
        svm_macro_f1_list, svm_micro_f1_list, nmi, ari = evaluate_results_nc(
            test_features.asnumpy(), instances_labels[test_idx],
            args.instance_classes)
        svm_macro_f1_lists.append(svm_macro_f1_list)
        svm_micro_f1_lists.append(svm_micro_f1_list)
        kmeans_nmi_lists.append(nmi)
        kmeans_ari_lists.append(ari)
    svm_macro_f1_lists = np.array(svm_macro_f1_lists).transpose()
    svm_micro_f1_lists = np.array(svm_micro_f1_lists).transpose()
    kmeans_nmi_lists = np.array(kmeans_nmi_lists)
    kmeans_ari_lists = np.array(kmeans_ari_lists)
    print('----------------------------------------------------------------')
    print('SVM:Macro-F1: ' + ', '.join([
        '{:.6f}~{:.6f} ({:.1f})'.format(np.mean(macro_f1), np.std(macro_f1),
                                        train_size)
        for macro_f1, train_size in zip(svm_macro_f1_lists, [0.6, 0.2])
    ]))
    print('SVM:Micro-F1: ' + ', '.join([
        '{:.6f}~{:.6f} ({:.1f})'.format(np.mean(micro_f1), np.std(micro_f1),
                                        train_size)
        for micro_f1, train_size in zip(svm_micro_f1_lists, [0.6, 0.2])
    ]))
    print('K-means:NMI: ' + '{:.6f}~{:.6f}'.format(np.mean(kmeans_nmi_lists),
                                                   np.std(kmeans_nmi_lists)))
    print('K-means:ARI: ' + '{:.6f}~{:.6f}'.format(np.mean(kmeans_ari_lists),
                                                   np.std(kmeans_ari_lists)))


if __name__ == '__main__':
    args.dataset = 'HIV'
    args.num_views = 2
    args.instance_classes = 2
    args.node_classes = 2
    args.save_postfix = 'RTGNN_HIV'
    start = time.time()
    RTGNN()
    end = time.time()
    print(end - start)
