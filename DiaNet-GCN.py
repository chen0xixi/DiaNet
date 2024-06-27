from __future__ import print_function
from gfnnutils import load_data
import argparse
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
from models1 import GCN1
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from graph_emb import GCN_AIR, GCN_edge
import scipy.sparse as sp
from early_stopping import EarlyStopping
from utils import accuracy, load_citation_edge

import networkx as nx
from sklearn.preprocessing import StandardScaler

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=5, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.002,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                    choices=['AugNormAdj'],
                    help='Normalization method for the adjacency matrix.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

warnings.filterwarnings("ignore")

path = "../data/"
dataset = "cora"
K = 10

adj, features, labels, idx_train, idx_val, idx_test, edge_adj, edge_name, T_matrix = load_citation_edge(dataset,
                                                                                                        args.normalization,
                                                                                                        K)
print(len(idx_train))
edge_name = torch.tensor(edge_name)
# Model and optimizer
model = GCN_AIR(n_n=features.shape[1],
                 nhid=args.hidden,
                 nclass=int(labels.max().item() + 1),
                 dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    T_matrix = T_matrix.cuda()
    edge_name = edge_name.cuda()
    edge_adj = edge_adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
print(adj.dtype, features.dtype, labels.dtype)

# adj, features, labels, idx_train, idx_val, idx_test = load_data(path, dataset)
loss_history = []
val_acc_history = []


def r_f(label, output_np):
    predicted_labels = np.argmax(output_np, axis=1)
    encoder = OneHotEncoder(sparse=False)
    # 拟合并转换矩阵
    one_hot_encoded = encoder.fit_transform(predicted_labels.reshape(-1, 1))
    sorted_indices = np.argsort(label)  # 获取按 label 升序排序的索引
    one_hot_encoded = one_hot_encoded[sorted_indices]
    hot = one_hot_encoded @ one_hot_encoded.T
    plt.figure(figsize=(10, 8))
    plt.imshow(hot, interpolation='bicubic', cmap='viridis')  # 使用双线性插值进行平滑
    plt.savefig('new1_cora_v.png', dpi=800)
    plt.title("Smoothed Heatmap")
    plt.show()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output1, output2, output3 = model(features, adj, edge_name, T_matrix, edge_adj)
    # if (epoch + 1) % 50 == 0 :
    #     visual(output[idx_train], labels[idx_train], epoch + 1, K, dataset, "GBSGCN")
    loss_train1 = F.cross_entropy(output1[idx_train], labels[idx_train])
    loss_train2 = F.cross_entropy(output2[idx_train], labels[idx_train])
    loss_train3 = F.cross_entropy(output3[idx_train], labels[idx_train])

    loss_train = (0.5 + 0.1 * 6) * loss_train1 + 0.4 * loss_train2 + 0.5 * loss_train3
    acc_train = accuracy(output1[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output1, output2, output3 = model(features, adj, edge_name, T_matrix, edge_adj)

    loss_val1 = F.cross_entropy(output1[idx_val], labels[idx_val])
    loss_val2 = F.cross_entropy(output2[idx_val], labels[idx_val])

    loss_val = 1.2 * loss_val1 + 0.4 * loss_val2

    acc_val = accuracy(output1[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    loss_history.append(loss_train.item())
    val_acc_history.append(acc_val.item())
    t = acc_val.item()
    # print('acc_val', t)
    return loss_val


def test():
    model.eval()
    output1, output2, output3 = model(features, adj, edge_name, T_matrix, edge_adj)

    loss_test1 = F.cross_entropy(output1[idx_test], labels[idx_test])
    loss_test2 = F.cross_entropy(output2[idx_test], labels[idx_test])
    '''
    ttt = output1.cpu().detach().numpy()
    r_f(labels.cpu().numpy(), ttt)
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(ttt)
    # 可视化降维后的数据
    plt.figure(figsize=(8, 6))
    v = labels.cpu().numpy()
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=v, cmap='rainbow')
    plt.colorbar(scatter)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.title('t-SNE Visualization')
    plt.savefig('new1_cora.png', dpi=800)
    plt.show()
    '''
    loss_test = 1.2 * loss_test1 + 0 * loss_test2

    acc_test = accuracy(output1[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return format(acc_test.item())


# Train model
t_total = time.time()

early_stopping = EarlyStopping(patience=2500, verbose=True)
for epoch in range(args.epochs):
    loss_val = train(epoch)
    early_stopping(loss_val, model)
    if early_stopping.early_stop:
        break
# model.load_state_dict(torch.load('checkpoint.pt'))
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
# Testing

test()
