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
import mindspore as ms
from mindspore import Tensor

np.set_printoptions(suppress=True)


def to_adjlist_DBLP(filename):
    file = open(filename, 'r')
    lines = file.readlines()[3:]
    file.close()
    adjlist = []
    weilist = np.zeros((len(lines), len(lines)))
    for i, line in enumerate(lines):
        tmp_list = [int(idx) for idx in line.strip().split(' ')]
        adjlist.append(tmp_list)
        for w in tmp_list:
            weilist[i][w] += 1
        weilist[i] = weilist[i] / np.max(weilist[i])
    # weilist = weilist.tolist()
    return adjlist, weilist


def load_data(dataset='HIV'):
    prefix = './data/preprocessed/' + dataset
    if dataset != 'DBLP':
        instances_labels = np.load(prefix + '/' + dataset + '_ins_labels.npy')
        feats_tensor = np.load(prefix + '/' + dataset + '_feat_tensor.npy')
        weighted_adjs_tensor = np.load(prefix + '/' + dataset +
                                       '_adj_tensor.npy')
        regions_labels = np.load(prefix + '/' + dataset + '_reg_labels.npy')
        train_val_test_idx = np.load(prefix + '/' + dataset +
                                     '_train_val_test.npz')
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
        return np.copy(
            self.indices[(self.iter_counter - 1) *
                         self.batch_size:self.iter_counter * self.batch_size])

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
    raw_data = sio.loadmat('../data/raw/' + dataset + '/' + dataset + '.mat')
    reg_labels = raw_data['reg_labels'].squeeze()
    np.save(
        '../data/preprocessed/' + dataset + '/' + dataset + '_reg_labels.npy',
        reg_labels)
    ins_labels = raw_data['label'].squeeze()
    ins_labels[ins_labels < 0] = 0
    np.save(
        '../data/preprocessed/' + dataset + '/' + dataset + '_ins_labels.npy',
        ins_labels)
    # !!!note: the fmri and dti views of the initial data have been exchanged
    raw_data['fmri'] = np.expand_dims(raw_data['fmri'].transpose(2, 0, 1), 1)
    raw_data['dti'] = np.expand_dims(raw_data['dti'].transpose(2, 0, 1), 1)
    adj_feat_tensor = np.concatenate((raw_data['fmri'], raw_data['dti']),
                                     axis=1)
    np.save(
        '../data/preprocessed/' + dataset + '/' + dataset + '_adj_tensor.npy',
        adj_feat_tensor)
    np.save(
        '../data/preprocessed/' + dataset + '/' + dataset + '_feat_tensor.npy',
        adj_feat_tensor)


def load_raw_data_DC(years_list, quarters_list, months_list):
    raw_stations_list = []
    for year in years_list[:3]:
        for quarter in quarters_list:
            filename = '../data/raw/BikeDC/' + year + quarter + '.csv'
            csv_file = pd.read_csv(filename)
            start_station = set(csv_file['Start station'])
            end_station = set(csv_file['End station'])
            raw_stations_list.append(start_station & end_station)
            print(filename)
    for year in years_list[3:]:
        for month in months_list:
            filename = '../data/raw/BikeDC/' + year + month + '.csv'
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
    raw_stations_name2id = {
        name: id
        for id, name in zip(range(len(raw_stations)), raw_stations)
    }
    print(len(raw_stations_name2id))
    print(raw_stations_name2id)
    np.save('../data/preprocessed/BikeDC/raw_stations_name2id.npy',
            raw_stations_name2id)


def get_loc_feats_DC():
    raw_stations_name2id = np.load(
        '../data/preprocessed/BikeDC/raw_stations_name2id.npy',
        allow_pickle=True)[()]
    filename = '../data/raw/BikeDC/2020/202012.csv'
    csv_file = pd.read_csv(filename)
    raw_stations_feats = csv_file[csv_file['start_station_name'].isin(
        raw_stations_name2id)]
    raw_stations_feats = raw_stations_feats[[
        'start_station_name', 'start_lat', 'start_lng'
    ]].drop_duplicates(subset=['start_station_name'], keep='first')
    raw_stations_feats = raw_stations_feats.replace(
        {"start_station_name": raw_stations_name2id})
    raw_stations_feats = raw_stations_feats.sort_values(
        by="start_station_name")
    raw_stations_feats = raw_stations_feats[['start_lat',
                                             'start_lng']].to_numpy()
    np.save('../data/preprocessed/BikeDC/raw_stations_feats.npy',
            raw_stations_feats)


# optional
def kmeans_distances_regions_DC(k=267):
    dataset = np.load('../data/preprocessed/BikeDC/raw_stations_feats.npy')
    # y_list = []
    # x_list = np.arange(50, 270, 10)
    # for x in x_list:
    # 	if x == 0:
    # 		x = 1
    # 	kmeans = KMeans(x).fit(dataset)
    # 	loss = kmeans.inertia_
    # 	y_list.append(loss)
    # plt.plot(x_list, y_list)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()
    kmeans = KMeans(k).fit(dataset)
    labelPred = kmeans.labels_
    centroids = kmeans.cluster_centers_
    loss = kmeans.inertia_
    print(f'labelPred:{labelPred}')
    print(f'centroids:{centroids}')
    print(f'loss:{loss}')
    res_regions = defaultdict(list)
    for i, id in enumerate(labelPred):
        res_regions[id].append(i)
    np.save('../data/preprocessed/BikeDC/res_regions.npy', res_regions)


# optional
def res_regions_DC():
    res_regions = np.load('../data/preprocessed/BikeDC/res_regions.npy',
                          allow_pickle=True)[()]
    raw_stations_name2id = np.load(
        '../data/preprocessed/BikeDC/raw_stations_name2id.npy',
        allow_pickle=True)[()]
    idold2new = {}
    for new_id, old_list in res_regions.items():
        for old_id in old_list:
            idold2new[old_id] = new_id
    res_regions_name2id = {}
    for name, old_id in raw_stations_name2id.items():
        res_regions_name2id[name] = idold2new[old_id]
    np.save('../data/preprocessed/res_regions_name2id.npy',
            res_regions_name2id)


def generate_adj_tensor_00_DC(view_list, years_list, quarters_list,
                              months_list):
    # if use res regions
    # n_num = len(np.load('../data/preprocessed/res_regions_name2id.npy',allow_pickle=True)[()].keys())
    n_num = np.load('../data/preprocessed/BikeDC/raw_stations_feats.npy',
                    allow_pickle=True).shape[0]
    BikeDC_adj_tensor = np.random.random((len(years_list) * len(months_list),
                                          len(view_list), n_num, n_num)) / 100.
    BikeDC_adj_tensor[BikeDC_adj_tensor < 0.005] = 0.
    # BikeDC_adj_tensor = np.zeros((len(years_list)*len(months_list), len(view_list), n_num, n_num))
    example_id = 0
    for i, year in enumerate(years_list):
        if i < 3:
            for quarter in quarters_list:
                filename = '../data/raw/BikeDC/' + year + quarter + '.csv'
                print(filename)
                csv_file = pd.read_csv(filename)
                months = [
                    i + (int(quarter.split('Q')[-1]) - 1) * 3
                    for i in range(1, 4)
                ]
                months = [
                    '0' + str(month) if month < 10 else str(month)
                    for month in months
                ]
                for month in months:
                    generate_adj_tensor_01_DC(year, month, csv_file,
                                              example_id, BikeDC_adj_tensor,
                                              n_num)
                    example_id += 1
        else:
            for month in months_list:
                filename = '../data/raw/BikeDC/' + year + month + '.csv'
                print(filename)
                csv_file = pd.read_csv(filename)
                generate_adj_tensor_01_DC(year, month, csv_file, example_id,
                                          BikeDC_adj_tensor, n_num)
                example_id += 1
    np.save('../data/preprocessed/BikeDC/BikeDC_adj_tensor.npy',
            BikeDC_adj_tensor)
    np.save('../data/preprocessed/BikeDC/BikeDC_feat_tensor.npy',
            BikeDC_adj_tensor)


def generate_adj_tensor_01_DC(year, month, csv_file, example_id,
                              BikeDC_adj_tensor, n_num):
    # if use res regions
    # res_regions_name2id = np.load('../data/preprocessed/BikeDC/res_regions_name2id.npy',allow_pickle=True)[()]
    raw_stations_name2id = np.load(
        '../data/preprocessed/BikeDC/raw_stations_name2id.npy',
        allow_pickle=True)[()]
    year_ = year.split('/')[-1]
    year_month = year_ + '-' + month
    if year_ == '2020' and int(month) >= 4:
        s_station = 'start_station_name'
        e_station = 'end_station_name'
        s_time = 'started_at'
    else:
        s_station = 'Start station'
        e_station = 'End station'
        s_time = 'Start date'
    weekdays = np.array(
        calendar.Calendar(calendar.MONDAY).monthdatescalendar(
            int(year_), int(month))[1:]).transpose()[:5].reshape(-1)
    weekends = np.array(
        calendar.Calendar(calendar.SATURDAY).monthdatescalendar(
            int(year_), int(month))[1:]).transpose()[:2].reshape(-1)
    weekdays = [m.strftime('%Y-%m-%d') for m in weekdays]
    weekends = [s.strftime('%Y-%m-%d') for s in weekends]
    days_list = [
        len(weekdays),
        len(weekends),
        calendar.monthrange(int(year_), int(month))[1]
    ]
    print(year_month, end='')
    print(days_list)
    target_elems = csv_file[csv_file[s_time].str.contains(year_month)]
    target_elems = target_elems[target_elems[s_station].isin(
        raw_stations_name2id)].replace({s_station: raw_stations_name2id})
    target_elems = target_elems[target_elems[e_station].isin(
        raw_stations_name2id)].replace({e_station: raw_stations_name2id})
    target_elems['Date'] = target_elems[s_time].str.split(' ', expand=True)[0]
    month_elems = target_elems[['Date', s_station, e_station]]
    weekday_elems = month_elems[month_elems['Date'].isin(weekdays)]
    weekend_elems = month_elems[month_elems['Date'].isin(weekends)]
    generate_adj_tensor_02_DC(BikeDC_adj_tensor, n_num, example_id,
                              weekday_elems, weekend_elems, month_elems,
                              s_station, e_station, days_list)


def generate_adj_tensor_02_DC(BikeDC_adj_tensor, n_num, example_id,
                              weekday_view, weekend_view, month_view,
                              s_station, e_station, days_list):
    for i, (start, end) in enumerate(
            zip(weekday_view[s_station], weekday_view[e_station])):
        BikeDC_adj_tensor[example_id][0][start][end] += 1
        BikeDC_adj_tensor[example_id][0][end][start] += 1
    for i, (start, end) in enumerate(
            zip(weekend_view[s_station], weekend_view[e_station])):
        BikeDC_adj_tensor[example_id][1][start][end] += 1
        BikeDC_adj_tensor[example_id][1][end][start] += 1
    for i, (start,
            end) in enumerate(zip(month_view[s_station],
                                  month_view[e_station])):
        BikeDC_adj_tensor[example_id][2][start][end] += 1
        BikeDC_adj_tensor[example_id][2][end][start] += 1
    for i, num_days in enumerate(days_list):
        BikeDC_adj_tensor[example_id][
            i] = BikeDC_adj_tensor[example_id][i] / num_days
        for j in range(n_num):
            BikeDC_adj_tensor[example_id][i][j] = BikeDC_adj_tensor[example_id][i][j]/\
                       np.sum(BikeDC_adj_tensor[example_id][i][j])


def label_regions_DC():
    raw_stations_feats = np.load(
        '../data/preprocessed/BikeDC/raw_stations_feats.npy')
    central_point = np.mean(raw_stations_feats, axis=0, keepdims=True)
    central_point = np.repeat(central_point,
                              axis=0,
                              repeats=len(raw_stations_feats))
    distances = raw_stations_feats - central_point
    distances = np.linalg.norm(distances, axis=1)
    sorted_indices = np.argsort(distances)
    l0 = sorted_indices[:math.ceil(len(distances) / 4)]
    l1 = sorted_indices[math.ceil(len(distances) /
                                  4):math.ceil(len(distances) / 2)]
    l2 = sorted_indices[math.ceil(len(distances) /
                                  2):math.ceil(len(distances) * 3 / 4)]
    l3 = sorted_indices[math.ceil(len(distances) * 3 / 4):]
    # l0_feats = raw_stations_feats[l0].transpose()
    # l1_feats = raw_stations_feats[l1].transpose()
    # l2_feats = raw_stations_feats[l2].transpose()
    # l3_feats = raw_stations_feats[l3].transpose()
    # l0_x, l0_y = l0_feats[0],l0_feats[1]
    # l1_x, l1_y = l1_feats[0],l1_feats[1]
    # l2_x, l2_y = l2_feats[0],l2_feats[1]
    # l3_x, l3_y = l3_feats[0],l3_feats[1]
    # plt.scatter(l0_x, l0_y, color='r')
    # plt.scatter(l1_x, l1_y, color='b')
    # plt.scatter(l2_x, l2_y, color='g')
    # plt.scatter(l3_x, l3_y, color='orange')
    # plt.show()
    BikeDC_reg_labels = np.empty(len(raw_stations_feats))
    BikeDC_reg_labels[l0] = 0
    BikeDC_reg_labels[l1] = 1
    BikeDC_reg_labels[l2] = 2
    BikeDC_reg_labels[l3] = 3
    BikeDC_reg_labels = BikeDC_reg_labels.astype(int)
    print(BikeDC_reg_labels)
    np.save('../data/preprocessed/BikeDC/BikeDC_reg_labels.npy',
            BikeDC_reg_labels)


def split_train_val_test(dataset, ratio_list):
    if dataset == 'HIV' or dataset == 'BP':
        raw_data = sio.loadmat('../data/raw/' + dataset + '/' + dataset +
                               '.mat')
        raw_data = raw_data['label'].squeeze()
        all_idx = [idx for idx in range(len(raw_data))]
        raw_data[raw_data < 0] = 0
    elif dataset == 'BikeDC':
        raw_data = np.array(
            [q for y in range(6) for q in range(4) for l in range(3)])
        all_idx = [idx for idx in range(72)]
        np.save('../data/preprocessed/BikeDC/BikeDC_ins_labels.npy', raw_data)
    else:
        raw_data = np.load(
            '../data/preprocessed/PROTEINS/PROTEINS_ins_labels.npy')
        all_idx = [idx for idx in range(1000)]
    shuffle(all_idx)
    train_idx = all_idx[:ratio_list[0]]
    val_idx = all_idx[ratio_list[0]:ratio_list[0] + ratio_list[1]]
    test_idx = all_idx[ratio_list[0] + ratio_list[1]:]
    np.savez('../data/preprocessed/' + dataset + '/' + dataset +
             '_train_val_test.npz',
             train=train_idx,
             val=val_idx,
             test=test_idx)
    print(raw_data[train_idx])
    print(raw_data[val_idx])
    print(raw_data[test_idx])
    print(ratio_list)
    print(np.sum(raw_data[train_idx]))
    print(np.sum(raw_data[val_idx]))
    print(np.sum(raw_data[test_idx]))


def load_raw_data_PROTEINS(dim_ins, dim_node):
    with open('../data/raw/PROTEINS/PROTEINS_node_labels.txt') as f:
        txt_file = f.read().split('\n')
        node_idx2label = np.empty((100000, ))
        num_node = 0
        for node_label in txt_file[:-1]:
            node_idx2label[num_node] = int(node_label)
            num_node += 1
        node_idx2label = node_idx2label[:num_node]
    with open('../data/raw/PROTEINS/PROTEINS_node_attributes.txt') as f:
        txt_file = f.read().split('\n')
        node_idx2feats = np.empty((num_node, dim_node))
        for node_idx, node_feats in enumerate(txt_file[:-1]):
            node_idx2feats[node_idx] = np.array(
                list(map(float, node_feats.split(','))))
    with open('../data/raw/PROTEINS/PROTEINS_graph_indicator.txt') as f:
        txt_file = f.read().split('\n')
        ins_idx2nodeidxes = {}
        for node_idx, ins_idx in enumerate(txt_file[:-1]):
            ins_idx = int(ins_idx) - 1
            if ins_idx not in ins_idx2nodeidxes.keys():
                ins_idx2nodeidxes[ins_idx] = [node_idx]
            else:
                ins_idx2nodeidxes[ins_idx].append(node_idx)
    with open('../data/raw/PROTEINS/PROTEINS_graph_labels.txt') as f:
        txt_file = f.read().split('\n')
        ins_idx2label = np.empty((len(ins_idx2nodeidxes.keys()), ))
        for ins_idx, graph_label in enumerate(txt_file[:-1]):
            ins_idx2label[ins_idx] = int(graph_label) - 1
    with open('../data/raw/PROTEINS/PROTEINS_A.txt') as f:
        txt_file = f.read().split('\n')
        ins_idx2edges = {}
        ins_idx = 0
        ins_idx2edges[ins_idx] = []
        tmp_num_node = len(ins_idx2nodeidxes[ins_idx])
        for edge in txt_file[:-1]:
            node_idx_start = int(edge.split(',')[0]) - 1
            node_idx_end = int(edge.split(',')[1]) - 1
            if node_idx_start >= tmp_num_node or node_idx_end >= tmp_num_node:
                ins_idx += 1
                ins_idx2edges[ins_idx] = []
                tmp_num_node += len(ins_idx2nodeidxes[ins_idx])
            ins_idx2edges[ins_idx].append((node_idx_start, node_idx_end))
        num_ins = 0
        PROTEINS_adj_tensor = np.empty(
            (len(ins_idx2edges.keys()), 2, dim_ins, dim_ins))
        PROTEINS_feat_tensor = np.empty(
            (len(ins_idx2edges.keys()), 2, dim_ins, dim_ins))
        PROTEINS_reg_labels = np.empty((len(ins_idx2edges.keys()), dim_ins),
                                       dtype=np.int)
        PROTEINS_ins_labels = np.empty((len(ins_idx2edges.keys()), ),
                                       dtype=np.int)
        for ins_idx, ins_edges in ins_idx2edges.items():
            node_idxes = ins_idx2nodeidxes[ins_idx]
            if len(node_idxes) <= dim_ins and num_ins < 1000:
                adj_matrix_view1, adj_matrix_view2, feat_matrix_view1, feat_matrix_view2 = \
                 generate_adj_tensor_PROTEINS(dim_ins, node_idxes, ins_edges, node_idx2feats[node_idxes])
                PROTEINS_adj_tensor[num_ins][0] = adj_matrix_view1
                PROTEINS_adj_tensor[num_ins][1] = adj_matrix_view2
                PROTEINS_feat_tensor[num_ins][0] = feat_matrix_view1
                PROTEINS_feat_tensor[num_ins][1] = feat_matrix_view2
                PROTEINS_reg_labels[num_ins] = np.array(
                    list(node_idx2label[node_idxes]) + [3] *
                    (dim_ins - len(node_idxes)))
                PROTEINS_ins_labels[num_ins] = np.array(ins_idx2label[ins_idx])
                num_ins += 1
        PROTEINS_adj_tensor = PROTEINS_adj_tensor[:num_ins]
        PROTEINS_feat_tensor = PROTEINS_feat_tensor[:num_ins]
        PROTEINS_reg_labels = PROTEINS_reg_labels[:num_ins]
        PROTEINS_ins_labels = PROTEINS_ins_labels[:num_ins]
        np.save('../data/preprocessed/PROTEINS/PROTEINS_adj_tensor.npy',
                PROTEINS_adj_tensor)
        np.save('../data/preprocessed/PROTEINS/PROTEINS_feat_tensor.npy',
                PROTEINS_feat_tensor)
        np.save('../data/preprocessed/PROTEINS/PROTEINS_reg_labels.npy',
                PROTEINS_reg_labels)
        np.save('../data/preprocessed/PROTEINS/PROTEINS_ins_labels.npy',
                PROTEINS_ins_labels)


def generate_adj_tensor_PROTEINS(dim_ins, node_idxes, ins_edges,
                                 node_idx2feats):
    node_old2newidx = {}
    node_idx_new = 0
    for node_idx_old in node_idxes:
        node_old2newidx[node_idx_old] = node_idx_new
        node_idx_new += 1
    adj_matrix_view1 = np.zeros((dim_ins, dim_ins), dtype=np.float)
    feat_matrix_view1 = np.zeros((dim_ins, dim_ins), dtype=np.float)
    # feat_matrix_view1 = np.random.random((dim_ins, dim_ins))/1000.
    # adj_matrix_view1 = np.random.random((dim_ins, dim_ins))/1000.
    for edge in ins_edges:
        node_idx_start = node_old2newidx[edge[0]]
        node_idx_end = node_old2newidx[edge[1]]
        adj_matrix_view1[node_idx_start][node_idx_end] = 1
        adj_matrix_view1[node_idx_end][node_idx_start] = 1
    # adj_matrix_view1 += np.eye(dim_ins)
    # feat_matrix_view1 = adj_matrix_view1
    adj_matrix_view2 = np.zeros((dim_ins, dim_ins), dtype=np.float)
    feat_matrix_view2 = np.zeros((dim_ins, dim_ins), dtype=np.float)
    # feat_matrix_view2 = np.random.random((dim_ins, dim_ins))/1000.
    # adj_matrix_view2_01 = np.random.random((dim_ins, dim_ins))/2000.
    for i in range(len(node_idxes)):
        for j in range(i + 1, len(node_idxes)):
            feat_start = node_idx2feats[i]
            feat_end = node_idx2feats[j]
            dist = np.sqrt(np.sum((feat_start - feat_end)**2))
            adj_matrix_view2[i][j] = dist
    adj_matrix_view2 += np.transpose(adj_matrix_view2)
    for i in range(len(node_idxes)):
        adj_matrix_view2[i, :len(node_idxes)] /= np.max(adj_matrix_view2)
        # feat_matrix_view1[i,:len(node_idxes)] = adj_matrix_view1[i,:len(node_idxes)]
        # feat_matrix_view2[i,:len(node_idxes)] = adj_matrix_view2[i,:len(node_idxes)]
        # feat_matrix_view1[i,len(node_idxes):] = np.random.random((1, dim_ins-len(node_idxes)))/1000.
        # feat_matrix_view2[i,len(node_idxes):] = np.random.random((1, dim_ins-len(node_idxes)))/1000.
    adj_matrix_view2 = np.ones(adj_matrix_view2.shape) - adj_matrix_view2
    adj_matrix_view2[adj_matrix_view2 >= 1.] = 0
    adj_matrix_view2[adj_matrix_view2 < 0.95] = 0
    # adj_matrix_view2 += np.eye(dim_ins)
    for i in range(len(node_idxes)):
        feat_matrix_view1[i, :len(node_idxes)] = adj_matrix_view1[
            i, :len(node_idxes)]
        feat_matrix_view2[i, :len(node_idxes)] = adj_matrix_view2[
            i, :len(node_idxes)]
    # feat_matrix_view1 += np.random.random((dim_ins, dim_ins))/2
    # feat_matrix_view2 += np.random.random((dim_ins, dim_ins))/10
    # feat_matrix_view2 += adj_matrix_view2
    # adj_matrix_view2 += np.eye(dim_ins) + np.random.random((dim_ins, dim_ins))/1000.
    return adj_matrix_view1, adj_matrix_view2, feat_matrix_view1, feat_matrix_view2


def data_preprocess(dataset='HIV'):
    if dataset == 'HIV' or dataset == 'BP':
        load_raw_data_brain(dataset)
        ratio_list = [24, 6, 40] if dataset == 'HIV' else [33, 8, 56]
        split_train_val_test(dataset, ratio_list)
    elif dataset == 'BikeDC':
        years_list = [
            str(year) + '/' + str(year) for year in range(2015, 2021)
        ]
        quarters_list = ['Q' + str(quarter) for quarter in range(1, 5)]
        months_list = [
            '0' + str(month) if month < 10 else str(month)
            for month in range(1, 13)
        ]
        view_list = ['weekday', 'weekend', 'month']
        ratio_list = [24, 6, 42]
        # load_raw_data_DC(years_list, quarters_list, months_list)
        # get_loc_feats_DC()
        # # optional
        # kmeans_distances_regions_DC()
        # res_regions_DC()
        generate_adj_tensor_00_DC(view_list, years_list, quarters_list,
                                  months_list)
        # label_regions_DC()
        split_train_val_test(dataset, ratio_list)
    elif dataset == 'PROTEINS':
        dim_ins = 80
        dim_node = 29
        load_raw_data_PROTEINS(dim_ins, dim_node)
        ratio_list = [300, 100, 600]
        # split_train_val_test(dataset, ratio_list)


if __name__ == '__main__':
    print('data processing...')
    # data_preprocess('HIV')
    # data_preprocess('BP')
    # data_preprocess('BikeDC')
    data_preprocess('PROTEINS')
    # The DBLP has been processed
