import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch
import numpy as np


class GCN_AIR(nn.Module):
    def __init__(self, n_n, nhid, nclass, dropout):
        super(GCN_AIR, self).__init__()

        self.gc1 = GraphConvolution(n_n, nhid)
        self.gc2 = GraphConvolution(n_n, nhid)
        self.gcADD = GraphConvolution(n_n, nhid)
        self.gc3 = GraphConvolution(3 * nhid, nclass)
        self.gc4 = GraphConvolution(nhid, nclass)
        self.gc5 = GraphConvolution(nhid, nclass)

        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(n_n, nhid)
        self.a = torch.nn.Parameter(torch.randn(1, 2 * nhid))

        self.dropout = dropout

    def forward(self, X_n, nadj, edge_name, T, eadj):
        X1 = self.gc1(X_n, nadj)
        X1 = F.dropout(X1, self.dropout, training=self.training)
        X2 = self.gc2(X_n, nadj)
        X2 = F.dropout(X2, self.dropout, training=self.training)
        # X_e = F.relu(torch.mul(X_n[edge_name[:, 0]], X_n[edge_name[:, 1]]))

        # X_e = F.relu((X_n[edge_name[:, 0]] + X_n[edge_name[:, 1]]))
        # X_e = F.relu(torch.mul(X1[edge_name[:, 0]], X2[edge_name[:, 1]]))
        X_e = F.relu((X1[edge_name[:, 0]] + X2[edge_name[:, 1]]))

        with torch.no_grad():
            X3 = T @ X_e

        X3 = X3 + 1 * torch.mul(X1, X2)
        X_e = F.relu(self.gc6(X_e, eadj))
        X_e = F.dropout(X_e, self.dropout, training=self.training)
        with torch.no_grad():
            X_e = T @ X_e
        X3 = X3 + 3 * X1
        result = torch.cat((X3, X1, X_e), dim=1)
        output1 = self.gc3(result, nadj)
        output1 = F.dropout(output1, self.dropout, training=self.training)
        output2 = self.gc4(X1, nadj)
        output2 = F.dropout(output2, self.dropout, training=self.training)
        output3 = self.gc4(X_e, nadj)
        output3 = F.dropout(output3, self.dropout, training=self.training)

        return F.log_softmax(output1, dim=1), F.log_softmax(output2, dim=1), F.log_softmax(output3, dim=1)


class GCN_edge(nn.Module):
    def __init__(self, n_n, nhid, nclass, dropout):
        super(GCN_edge, self).__init__()
        self.gc1 = GraphConvolution(n_n, nhid)
        self.gc2 = GraphConvolution(n_n, nhid)

        self.gc3 = GraphConvolution(3 * nhid, nclass)
        self.line1 = nn.Linear(nhid, nhid)
        self.line2 = nn.Linear(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nclass)
        self.gc5 = GraphConvolution(nhid, nclass)

        self.gc6 = GraphConvolution(nhid, nhid)

        self.dropout = dropout

    def forward(self, X_n, nadj, edge_name, T, eadj):
        X1 = self.gc1(X_n, nadj)
        X1 = F.dropout(X1, self.dropout, training=self.training)
        X2 = self.gc2(X_n, nadj)
        X2 = F.dropout(X2, self.dropout, training=self.training)
        X_e = F.relu(torch.mul(X1[edge_name[:, 0]], X1[edge_name[:, 1]]))
        # X_e = F.relu((X1[edge_name[:, 0]] + X1[edge_name[:, 1]]))
        with torch.no_grad():
            X3 = T @ X_e
        X3 = self.line1(X3)
        X3 = F.dropout(X3, self.dropout, training=self.training)
        X3 = X3 + 1 * torch.mul(X1, X2)
        X_e = F.relu(self.gc6(X_e, eadj))
        X_e = F.dropout(X_e, self.dropout, training=self.training)
        with torch.no_grad():
            X_e = T @ X_e
        X_e = self.line2(X_e)
        X_e = F.dropout(X_e, self.dropout, training=self.training)
        X3 = X3 + 2 * X1
        result = torch.cat((X3, X1, X_e), dim=1)

        output1 = self.gc3(result, nadj)
        output1 = F.dropout(output1, self.dropout, training=self.training)
        output2 = self.gc4(X1, nadj)
        output2 = F.dropout(output2, self.dropout, training=self.training)
        output3 = self.gc4(X_e, nadj)
        output3 = F.dropout(output3, self.dropout, training=self.training)

        return F.log_softmax(output1, dim=1), F.log_softmax(output2, dim=1), F.log_softmax(output3, dim=1)
