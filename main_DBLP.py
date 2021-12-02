import argparse
import torch
import numpy as np
from toolbox.data_preprocess import *
from toolbox.early_stopping import *
from toolbox.evaluation import *
from model.RTGNN_DBLP import OneLayerRTGNN_DBLP
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ap = argparse.ArgumentParser(description='RTGNN-DBLP')
ap.add_argument('--num-views', type=int, default=3, help='Number of views.')
ap.add_argument('--instance-classes', type=int, default=4, help='Number of instance types.')
ap.add_argument('--node-classes', type=int, default=4, help='Number of node types.')
ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the feature matrix.')
ap.add_argument('--inter-type', default='gcn', help='Types of inter gnns.')
ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector.')
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
ap.add_argument('--batch-size', type=int, default=64, help='Batch size.')
ap.add_argument('--patience', type=int, default=5, help='Patience.')
ap.add_argument('--save-postfix', default='RTGNN_DBLP', help='Postfix for the saved model and result.')
args = ap.parse_args()

def RTGNN_DBLP():
    print('RTGNN_DBLP...')
    _, weilist, features, instances_labels, train_val_test_idx = load_data('DBLP')
    weilist = np.vstack((np.expand_dims(weilist[0],0),np.expand_dims(weilist[1],0),np.expand_dims(weilist[2],0))) #(3, 4057, 4057)
    weilist = torch.FloatTensor(weilist)
    features = torch.FloatTensor(features)
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.sort(test_idx)
    svm_macro_f1_lists = []
    svm_micro_f1_lists = []
    for _ in range(args.repeat):
        model = OneLayerRTGNN_DBLP(features, weilist,
                               args.num_views, args.instance_classes, args.node_classes, args.hidden_dim,
                               args.dropout, args.slope,
                               args.RL_step, args.RL_start, args.threshold_start,
                               args.lambeda,
                               args.inter_type, args.attn_vec_dim)
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
                train_loss, _ = model.loss((batch_train_idx, batch_train_labels, True, epoch, iter, train_num_batchs))
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                if iter % train_num_batchs ==0:
                    print('Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f}'.format(epoch, iter, train_loss.item()))
            model.eval()
            with torch.no_grad():
                val_num_batchs = val_idx_generator.num_iterations()
                for _ in range(val_num_batchs):
                    batch_val_idx = val_idx_generator.next()
                    batch_val_labels = instances_labels[batch_val_idx]
                    val_loss, _ = model.loss((batch_val_idx, batch_val_labels, False, _, _, _))
            print('Epoch {:05d} | Val_Loss {:.4f} '.format(epoch, val_loss.item()))
            model_stop(val_loss, model)
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
                _, test_features = model.loss((batch_test_idx, batch_test_labels, False, _, _, _))
                test_features_list.append(test_features)
            test_features = torch.cat(test_features_list, 0)
            svm_macro_f1_list, svm_micro_f1_list = evaluate_results_nc_DBLP(test_features.cpu().numpy(), instances_labels[test_idx])
        svm_macro_f1_lists.append(svm_macro_f1_list)
        svm_micro_f1_lists.append(svm_micro_f1_list)
    svm_macro_f1_lists = np.array(svm_macro_f1_lists).transpose()
    svm_micro_f1_lists = np.array(svm_micro_f1_lists).transpose()
    print('----------------------------------------------------------------')
    print('SVM:Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        np.mean(macro_f1), np.std(macro_f1), train_size) for macro_f1, train_size in
        zip(svm_macro_f1_lists, [0.6, 0.2])]))
    print('SVM:Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        np.mean(micro_f1), np.std(micro_f1), train_size) for micro_f1, train_size in
        zip(svm_micro_f1_lists, [0.6, 0.2])]))

if __name__ == '__main__':
    start = time.time()
    RTGNN_DBLP()
    end = time.time()
    print(end-start)
    