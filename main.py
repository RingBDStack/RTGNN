import argparse
import torch
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ap = argparse.ArgumentParser(description='RTGNN')
ap.add_argument('--dataset', default='HIV', help='Dataset name')
ap.add_argument('--num-views', type=int, default=2, help='Number of views.')
ap.add_argument('--instance-classes', type=int, default=2, help='Number of instance types.')
ap.add_argument('--node-classes', type=int, default=2, help='Number of node types.')
ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the feature matrix.')
ap.add_argument('--inter-type', default='gcn', help='Types of inter gnns.')
ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector.')
ap.add_argument('--mat2vec', default='mean', help='Vectorization.')
ap.add_argument('--dropout', type=float, default=0.5, help='Dropout.')
ap.add_argument('--slope', type=float, default=0.2, help='The slope of Leaky Relu')
ap.add_argument('--lr', type=float, default=0.005, help='Learning rate.')
ap.add_argument('--weight_decay', type=float, default=0.001, help='The weight decay of the optimizer.')
ap.add_argument('--lambeda', type=float, default=1.0, help='Edge loss weight.')
ap.add_argument('--RL-step', type=float, default=0.02, help='Action step size of reinforcement learning.')
ap.add_argument('--RL-start', type=int, default=2, help='The epoch at which reinforcement learning begins.')
ap.add_argument('--threshold-start', default=0.5, help='The initial thresholds.')
ap.add_argument('--repeat', type=int, default=20, help='Repeat the training and testing for N times.')
ap.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
ap.add_argument('--batch-size', type=int, default=4, help='Batch size.')
ap.add_argument('--patience', type=int, default=5, help='Patience.')
ap.add_argument('--save-postfix', default='RTGNN_HIV', help='Postfix for the saved model and result.')
args = ap.parse_args()

def plt_hot_map(adj_tensor, dataset):
    colormap = LinearSegmentedColormap.from_list("", ['floralwhite', 'darkorange',
                                                      'tomato', 'orangered', 'red',
                                                      'firebrick', 'darkred'])
    plt.figure(figsize=(7, 6))
    h = sns.heatmap(
        data=adj_tensor,
        cmap=colormap,
        annot=False,
        cbar=False,
        vmax=1.0,
        vmin=0.0,
        xticklabels=20,
        yticklabels=20)
    cb = h.figure.colorbar(h.collections[0])  # show colorbar
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

def generate_reliable_adj_tensor(dataset,init_adjs_tensor):
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
                tmp_matrix[i][np.argpartition(imp_matrix[i], -int(N/2))[-int(N/2):]] = 1
                tmp_matrix[i][tmp_matrix[i] != 1] = 0
            tmp_matrix = tmp_matrix + np.transpose(tmp_matrix) + np.eye(N)
            tmp_matrix[tmp_matrix > 1] = 1
            for i in range(N):
                for j in range(N):
                    if tmp_matrix[i][j] == 1:
                        adj_matrix[i][j] = np.exp(-(1 / 2) * np.linalg.norm(U[j] - U[i], ord=2))
                    feat_matrix[i][j] = adj_matrix[i][j]
                adj_matrix[i] = adj_matrix[i] / np.max(adj_matrix[i])
                feat_matrix[i] = feat_matrix[i] / np.max(feat_matrix[i])
                reli_adj_tensor[m, v, i] = adj_matrix[i]
                init_feat_tensor[m, v, i] = feat_matrix[i]
    # plt_hot_map(reli_adj_tensor[0][0], args.dataset)
    # plt_hot_map(reli_adj_tensor[0][1], args.dataset)
    # reli_feat_tensor,_ = HOSVD_01(init_feat_tensor)
    # plt_hot_map(reli_feat_tensor[0][0], args.dataset)
    # plt_hot_map(reli_feat_tensor[0][1], args.dataset)
    return init_feat_tensor, reli_adj_tensor

def RTGNN():
    print('RTGNN_{}...'.format(args.dataset))
    feats_tensor, weighted_adjs_tensor, regions_labels, instances_labels, train_val_test_idx = load_data(args.dataset)
    if args.dataset == 'HIV' or args.dataset == 'BP':
        feats_tensor, weighted_adjs_tensor = generate_reliable_adj_tensor(args.dataset,weighted_adjs_tensor)
    G, _ = HOSVD_01(feats_tensor)
    # G += feats_tensor
    features = torch.FloatTensor(G)
    # features = torch.FloatTensor(feats_tensor)
    weighted_adjs = torch.FloatTensor(weighted_adjs_tensor)
    train_idx = train_val_test_idx['train']
    val_idx = train_val_test_idx['val']
    test_idx = train_val_test_idx['test']
    svm_macro_f1_lists = []
    svm_micro_f1_lists = []
    kmeans_nmi_lists = []
    kmeans_ari_lists = []
    for _ in range(args.repeat):
        model = OneLayerRTGNN(args.dataset,
                              features, weighted_adjs,
                              args.num_views, args.instance_classes, args.node_classes, args.hidden_dim,
                              args.dropout, args.slope,
                              args.RL_step, args.RL_start, args.threshold_start,
                              args.lambeda,
                              args.inter_type, args.attn_vec_dim, args.mat2vec)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model_stop = early_stopping(patience=args.patience, verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(args.save_postfix))
        train_idx_generator = index_generator(batch_size=args.batch_size, indices=train_idx)
        val_idx_generator = index_generator(batch_size=args.batch_size, indices=val_idx, shuffle=False)
        for epoch in range(args.epochs):
            model.train()
            train_num_batchs = train_idx_generator.num_iterations()
            for iter in range(train_num_batchs):
                batch_train_idx = train_idx_generator.next()
                batch_train_labels = instances_labels[batch_train_idx]
                train_loss, _ = model.loss((batch_train_idx, batch_train_labels, regions_labels, True, epoch, iter, train_num_batchs))
                optimizer.zero_grad()
                (train_loss[0]+train_loss[1]).backward()
                optimizer.step()
                if iter % train_num_batchs ==0:
                    print('Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f}'.format(epoch, iter, train_loss[0].item()))
            model.eval()
            with torch.no_grad():
                val_num_batchs = val_idx_generator.num_iterations()
                for _ in range(val_num_batchs):
                    batch_val_idx = val_idx_generator.next()
                    batch_val_labels = instances_labels[batch_val_idx]
                    val_loss, _ = model.loss((batch_val_idx, batch_val_labels, regions_labels, False, _, _, _))
            print('Epoch {:05d} | Val_Loss {:.4f} '.format(epoch, val_loss[0].item()))
            model_stop(val_loss[0], model)
            if model_stop.early_stop:
                print('Early stopping!')
                break
        model.eval()
        model.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(args.save_postfix)))
        test_idx_generator = index_generator(batch_size=args.batch_size, indices=test_idx, shuffle=False)
        test_features_list = []
        with torch.no_grad():
            test_num_batchs = test_idx_generator.num_iterations()
            for _ in range(test_num_batchs):
                batch_test_idx = test_idx_generator.next()
                batch_test_labels = instances_labels[batch_test_idx]
                _, test_features = model.loss((batch_test_idx, batch_test_labels, regions_labels, False, _, _, _))
                test_features_list.append(test_features)
            test_features = torch.cat(test_features_list, 0)
            svm_macro_f1_list, svm_micro_f1_list, nmi, ari = evaluate_results_nc(test_features.cpu().numpy(), instances_labels[test_idx], args.instance_classes)
        svm_macro_f1_lists.append(svm_macro_f1_list)
        svm_micro_f1_lists.append(svm_micro_f1_list)
        kmeans_nmi_lists.append(nmi)
        kmeans_ari_lists.append(ari)
    svm_macro_f1_lists = np.array(svm_macro_f1_lists).transpose()
    svm_micro_f1_lists = np.array(svm_micro_f1_lists).transpose()
    kmeans_nmi_lists = np.array(kmeans_nmi_lists)
    kmeans_ari_lists = np.array(kmeans_ari_lists)
    print('----------------------------------------------------------------')
    print('SVM:Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        np.mean(macro_f1), np.std(macro_f1), train_size) for macro_f1, train_size in
        zip(svm_macro_f1_lists, [0.6, 0.2])]))
    print('SVM:Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        np.mean(micro_f1), np.std(micro_f1), train_size) for micro_f1, train_size in
        zip(svm_micro_f1_lists, [0.6, 0.2])]))
    print('K-means:NMI: ' + '{:.6f}~{:.6f}'.format(np.mean(kmeans_nmi_lists),np.std(kmeans_nmi_lists)))
    print('K-means:ARI: ' + '{:.6f}~{:.6f}'.format(np.mean(kmeans_ari_lists),np.std(kmeans_ari_lists)))

if __name__ == '__main__':
    # args.dataset = 'BikeDC'
    # args.num_views = 3
    # args.instance_classes = 4
    # args.node_classes = 4
    # args.save_postfix = 'RTGNN_BikeDC'
    # start = time.time()
    # RTGNN()
    # end = time.time()
    # print(end - start)

    args.dataset = 'HIV'
    args.num_views = 2
    args.instance_classes = 2
    args.node_classes = 2
    args.save_postfix = 'RTGNN_HIV'
    start = time.time()
    RTGNN()
    end = time.time()
    print(end-start)

    # args.dataset = 'BP'
    # args.num_views = 2
    # args.instance_classes = 2
    # args.node_classes = 2
    # args.save_postfix = 'RTGNN_BP'
    # start = time.time()
    # RTGNN()
    # end = time.time()
    # print(end-start)

    # args.dataset = 'PROTEINS'
    # args.num_views = 2
    # args.instance_classes = 2
    # args.node_classes = 4
    # args.save_postfix = 'RTGNN_PROTEINS'
    # start = time.time()
    # RTGNN()
    # end = time.time()
    # print(end-start)