import math
import sys
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F

from normalization import fetch_normalization, row_normalize


def get_balls(features, labels, purity):
    X_train = features
    Y_train = labels
    init_l = GBS.main(X_train, Y_train, purity)
    return init_l


def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    # 坐标系ax1画曲线1
    ax1 = fig.add_subplot(111)  # 指的是将plot界面分成1行1列，此子图占据从左到右从上到下的1位置
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)  # c为颜色
    plt.ylabel('Loss')

    # 坐标系ax2画曲线2
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)  # 其本质就是添加坐标系，设置共享ax1的x轴，ax2背景透明
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()  # 开启右边的y坐标

    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')
    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()


def load_data2(data, labels):
    edges_unordered = np.genfromtxt("{}{}.cites".format("../data/cora/", "cora"),
                                    dtype=np.int32)

    index = data[:, -1]
    index_num = index.shape[0]
    idx = np.array(index, dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    edge_num = edges_unordered.shape[0]
    data_num = data.shape[0]
    DataAll = np.empty(shape=[0, 2])
    edge = DataAll

    for i in range(int(data_num)):
        for j in range(int(edge_num)):
            if data[i - 1, -1] == edges_unordered[j - 1, 0]:
                DataAll = np.vstack((DataAll, edges_unordered[j - 1, :]))
            if data[i - 1, -1] == edges_unordered[j - 1, 1]:
                DataAll = np.vstack((DataAll, edges_unordered[j - 1, :]))

    edge1_num = DataAll.shape[0]
    for m in range(int(edge1_num)):
        flag = 0
        for n in range(int(index_num)):
            if DataAll[m - 1, 0] == index[n - 1]:
                for s in range(int(index_num)):
                    if DataAll[m - 1, 1] == index[s - 1]:
                        flag = 1
        if flag == 1:
            edge = np.vstack((edge, DataAll[m - 1, :]))

    edges = np.array(list(map(idx_map.get, edge.flatten())),
                     dtype=np.int32).reshape(edge.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = data[:, 0: -2]
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels


def load_data23(data, labels):
    edges_unordered = np.genfromtxt("{}{}.cites".format("../data/cora/", "cora"),
                                    dtype=np.int32)

    index = data[0, :, -1]
    index_num = index.shape[0]
    idx = np.array(index, dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    edge_num = edges_unordered.shape[0]
    data_num = data.shape[1]
    DataAll = np.empty(shape=[0, 2])
    edge = DataAll

    for i in range(int(data_num)):
        for j in range(int(edge_num)):
            if data[0, i - 1, -1] == edges_unordered[j - 1, 0]:
                DataAll = np.vstack((DataAll, edges_unordered[j - 1, :]))
            if data[0, i - 1, -1] == edges_unordered[j - 1, 1]:
                DataAll = np.vstack((DataAll, edges_unordered[j - 1, :]))

    edge1_num = DataAll.shape[0]
    for m in range(int(edge1_num)):
        flag = 0
        for n in range(int(index_num)):
            if DataAll[m - 1, 0] == index[n - 1]:
                for s in range(int(index_num)):
                    if DataAll[m - 1, 1] == index[s - 1]:
                        flag = 1
        if flag == 1:
            edge = np.vstack((edge, DataAll[m - 1, :]))

    edges = np.array(list(map(idx_map.get, edge.flatten())),
                     dtype=np.int32).reshape(edge.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = torch.tensor(data[:, :, 0: -2]).float()

    for i in range(features.shape[0]):
        features[i, :, :] = F.normalize(features[i, :, :], dim=0)

    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def load_data1(data, labels):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("F:/GCN OWN/data/ind.{}.{}".format("cora".lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    features = data[:, 0: -2]
    features = normalize(features)
    num_data = data.shape[0]
    index = data[:, -1]
    adj_all = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    edge = np.empty(shape=[0, 2])

    for i in range(num_data):
        a = int(index[i])
        for j in range(num_data):
            b = int(index[j])
            if adj_all[a, b] == 1:
                edge = np.vstack((edge, [a, b]))

    idx = np.array(index, dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edge.flatten())),
                     dtype=np.int32).reshape(edge.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, size=shape)  # torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


def load_citation(dataset_str="cora", normalization="AugNormAdj"):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("F:/GCN OWN/data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("F:/GCN OWN/data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)

    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()


def create_edge_adj(vertex_adj):
    vertex_adj.setdiag(0)
    edge_index = np.nonzero(sp.triu(vertex_adj, k=1))
    num_edge = int(len(edge_index[0]))
    edge_name = [x for x in zip(edge_index[0], edge_index[1])]

    edge_adj = np.zeros((num_edge, num_edge))
    for i in range(num_edge):
        for j in range(i, num_edge):
            if len(set(edge_name[i]) & set(edge_name[j])) == 0:
                edge_adj[i, j] = 0
            else:
                edge_adj[i, j] = 1
    adj = edge_adj + edge_adj.T
    np.fill_diagonal(adj, 1)
    return sp.csr_matrix(adj), edge_name


def create_transition_matrix(vertex_adj):
    vertex_adj.setdiag(0)
    edge_index = np.nonzero(sp.triu(vertex_adj, k=1))
    num_edge = int(len(edge_index[0]))
    edge_name = [x for x in zip(edge_index[0], edge_index[1])]

    row_index = [i for sub in edge_name for i in sub]
    col_index = np.repeat([i for i in range(num_edge)], 2)

    data = np.ones(num_edge * 2)
    T = sp.csr_matrix((data, (row_index, col_index)),
                      shape=(vertex_adj.shape[0], num_edge))

    return T


def load_citation_edge(dataset="cora", normalization="AugNormAdj", K=5):
    # 这里是提取数据的函数，其中需要每一次实验进行更改。
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    # 引文数据格式中的所有单位
    objects = []

    for i in range(len(names)):
        with open("E:/pythonProject3/GBSGCN/GBSGCN/data/ind.{}.{}".format(dataset.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    # 如果python中版本较低，那么将改一下提取语句，
    # latin1Latin-1（也称为 ISO-8859-1）是一种8位编码，涵盖了大多数西欧语言的字符集。
    # 它是由国际标准化组织（ISO）定义的编码方式之一，也是最早的一种字符编码方案之一。

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    # 邻接矩阵adj

    T_matrix = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    T_matrix = create_transition_matrix(T_matrix)

    edge_adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    edge_adj, edge_name = create_edge_adj(edge_adj)
    edge_adj = normalize_adj(edge_adj + sp.eye(edge_adj.shape[0]))
    # 邻接矩阵adj

    test_idx_reorder = parse_index_file("E:/pythonProject3/GBSGCN/data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)
    empetySet = []

    if dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))

    T_matrix = torch.FloatTensor(np.array(T_matrix.todense()))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    no_class = set(np.where(labels)[1])

    def missing_elements(L):
        start, end = L[0], L[-1]
        return sorted(set(range(start, end + 1)).difference(L))

    # if dataset == 'citeseer':
    #     save_label = np.where(labels)[1]
    #     L = np.sort(test_idx_reorder)
    #     missing = missing_elements(L)
    #     for element in missing:
    #         save_label = np.insert(save_label, element, 0)
    #     labels = torch.FloatTensor(save_label)
    # else:
    #     labels = torch.FloatTensor(np.where(labels)[1])

    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    idx = np.arange(len(labels))
    idx = list(set(idx).difference(empetySet))
    np.random.shuffle(idx)
    train_size = math.ceil(len(labels) * K / 100 / len(no_class))
    train_size = [train_size for i in range(len(no_class))]

    idx_train = []
    count = [0 for i in range(len(no_class))]
    label_each_class = train_size
    next = 0

    for i in idx:
        if count == label_each_class:
            break
        next += 1
        for j in range(len(no_class)):
            if j == labels[i] and count[j] < label_each_class[j]:
                idx_train.append(i)
                count[j] += 1

    idx_val = idx[next:next + 500]
    test_size = len(test_idx_range.tolist())
    idx_test = idx[-test_size:] if test_size else idx[next:]

    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    edge_adj = sparse_mx_to_torch_sparse_tensor(edge_adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, edge_adj, edge_name, T_matrix


def load_citation1(dataset="cora", normalization="AugNormAdj", K=5):
    # 这里是提取数据的函数，其中需要每一次实验进行更改。
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    # 引文数据格式中的所有单位
    objects = []

    for i in range(len(names)):
        with open("E:/pythonProject3/GBSGCN/GBSGCN/data/ind.{}.{}".format(dataset.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    # 如果python中版本较低，那么将改一下提取语句，
    # latin1Latin-1（也称为 ISO-8859-1）是一种8位编码，涵盖了大多数西欧语言的字符集。
    # 它是由国际标准化组织（ISO）定义的编码方式之一，也是最早的一种字符编码方案之一。

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    # 邻接矩阵adj

    test_idx_reorder = parse_index_file("E:/pythonProject3/GBSGCN/data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)
    empetySet = []

    if dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    no_class = set(np.where(labels)[1])

    def missing_elements(L):
        start, end = L[0], L[-1]
        return sorted(set(range(start, end + 1)).difference(L))

    # if dataset == 'citeseer':
    #     save_label = np.where(labels)[1]
    #     L = np.sort(test_idx_reorder)
    #     missing = missing_elements(L)
    #     for element in missing:
    #         save_label = np.insert(save_label, element, 0)
    #     labels = torch.FloatTensor(save_label)
    # else:
    #     labels = torch.FloatTensor(np.where(labels)[1])

    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    idx = np.arange(len(labels))
    idx = list(set(idx).difference(empetySet))
    np.random.shuffle(idx)
    train_size = math.ceil(len(labels) * K / 100 / len(no_class))
    train_size = [train_size for i in range(len(no_class))]

    idx_train = []
    count = [0 for i in range(len(no_class))]
    label_each_class = train_size
    next = 0

    for i in idx:
        if count == label_each_class:
            break
        next += 1
        for j in range(len(no_class)):
            if j == labels[i] and count[j] < label_each_class[j]:
                idx_train.append(i)
                count[j] += 1

    idx_val = idx[next:next + 500]
    test_size = len(test_idx_range.tolist())
    idx_test = idx[-test_size:] if test_size else idx[next:]

    adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def visual(output, train_labels, epoch, K, dataset, methodname) -> object:
    output = output.to('cpu').detach().numpy()
    train_labels = train_labels.to('cpu').detach().numpy()
    output = TSNE(n_components=2).fit_transform(output)
    figure = plt.figure(figsize=(5, 5))

    color_idx = {}
    for i in range(output.shape[0]):
        color_idx.setdefault(train_labels[i], [])
        color_idx[train_labels[i]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(output[idx, 0], output[idx, 1], label=c, s=8)  # s=8是点的大小
    plt.legend(loc=2)
    plt.savefig('F:/GCN OWN/result/' + str(K) + '_' + dataset + '_' + methodname + '_' + str(epoch) + '.png',
                dpi=1000)  # dpi=1000是点的像素


def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    # 坐标系ax1画曲线1
    ax1 = fig.add_subplot(111)  # 指的是将plot界面分成1行1列，此子图占据从左到右从上到下的1位置
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)  # c为颜色
    plt.ylabel('Loss')

    # 坐标系ax2画曲线2
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)  # 其本质就是添加坐标系，设置共享ax1的x轴，ax2背景透明
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()  # 开启右边的y坐标

    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()


def load_txt_file(adj_data_set, features_set):
    features_set = pd.read_csv(features_set, sep='\t')
    features_set = features_set.iloc[:, :].to_numpy()
    features = features_set[:, 1:features_set.shape[1] - 1]
    features = np.asarray([list(map(int, x.split(','))) for sublist in features for x in sublist])
    features = torch.tensor(features, dtype=torch.float32)

    labels = features_set[:, -1].astype(np.float32)
    labels = torch.tensor(labels, dtype=torch.int64)

    adj_sp = pd.read_csv(adj_data_set, sep='\t')
    adj_sp = adj_sp.iloc[1::, :].to_numpy()
    adj = np.zeros((labels.shape[0], labels.shape[0]))
    print(labels.shape[0])
    for i in range(adj_sp.shape[0]):
        adj[adj_sp[i, 0], adj_sp[i, 1]] = 1
    adj = csr_matrix(adj, dtype=np.float32)
    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()
    adj = torch.tensor(adj).float()

    set_num = [140, 40, 1000]
    random_numbers = np.random.choice(labels.shape[0], size=sum(set_num), replace=False)
    idx_train, idx_val, idx_test = random_numbers[0:set_num[0]], random_numbers[
                                                                 set_num[0]:set_num[1] + set_num[0]], random_numbers[
                                                                                                      set_num[1] +
                                                                                                      set_num[0]:
                                                                                                      sum(set_num)]
    idx_train, idx_val, idx_test = torch.tensor(idx_train), torch.tensor(idx_val), torch.tensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test
