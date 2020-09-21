import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import to_torch_sparse_tensor, to_list

class AutoGNN(nn.Module):

    def __init__(self, feat_dim, graph,
                 hidden_dim, step_num, sample_num, nonlinear, agg,
                 class_num, dropout):
        super(AutoGNN, self).__init__()

        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.sample_num = sample_num
        self.step_num = step_num
        self.nonlinear = nonlinear
        self.agg = agg
        self.class_num = class_num

        self.sampleGraph(graph)
        self.precompute = False

        self.W =  nn.ModuleList([nn.Linear(feat_dim, hidden_dim, bias=False)])
        for _ in range(step_num-2):
            self.W.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.lastW = nn.Linear(hidden_dim, class_num, bias=False)
        self.precomputeW = nn.Linear(feat_dim, class_num, bias=False)

        self.drop = dropout > 0 and dropout < 1
        if self.drop:
            self.Dropout = nn.Dropout(dropout)

        for w in self.W:
            nn.init.xavier_uniform_(w.weight.data)

    def sampleGraph(self, graph):
        if self.sample_num == -1:
            self.adj = to_torch_sparse_tensor(graph)
            return

        self.adj={}
        for i in range(len(graph)):
            self.adj[i] = []
        for node,neighbors in graph.items():
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.sample_num:
                neighbors = np.random.choice(neighbors, self.sample_num, replace=False)
            elif len(neighbors) < self.sample_num:
                neighbors = np.random.choice(neighbors, self.sample_num, replace=True)
            self.adj[node] = neighbors
        self.adj = to_torch_sparse_tensor(self.adj, self.agg)

    def initWeight(self, initW):
        for w in self.W:
            w.weight.data = initW
            for param in w.parameters():
                param.requires_grad = False

    def precomputeX(self, X):
        if self.nonlinear == False:
            for _ in range(self.step_num):
                X = torch.spmm(self.adj, X)
            self.precompX = X
            self.precompute = True

    def forward(self, X):
        if self.precompute:
            X = self.precomputeW(self.precompX)
        else:
            for w in self.W:
                X = torch.spmm(self.adj, X)
                X = w(X)
                if self.drop:
                    X = self.Dropout(X)
                if self.nonlinear:
                    X = F.relu(X)
            X = self.lastW(X)
        return X

def get_auto_model(feat_dim, graph,
                   hidden_dim, step_num, sample_num, nonlinear, agg,
                   class_num, dropout):
    model = AutoGNN(feat_dim, graph,
                    hidden_dim, step_num, sample_num, nonlinear, agg,
                    class_num, dropout)
    return model

