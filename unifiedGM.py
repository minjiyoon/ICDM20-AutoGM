import torch
import torch.nn.functional as F
import torch.optim as optim

from args import get_args
from models import get_model, get_auto_model
from utils import load_citation, load_inductive_dataset

import numpy as np
from time import perf_counter
from sklearn import metrics
from math import log

dim_coff = 10e-19

class UnifiedGM():
    def __init__(self):
        self.load_data()

    def load_data(self):
        self.args = get_args()
        self.args.inductive = self.args.dataset in ['reddit', 'ppi']
        self.args.sigmoid = self.args.dataset in ['ppi']

        start_time = perf_counter()
        if not self.args.inductive:
            self.graph, self.feats, self.labels, self.label_max, self.idx, _ = load_citation(self.args.dataset)
        else:
            self.graph, self.graph_test, self.feats, self.labels, self.label_max, self.idx = load_inductive_dataset(self.args.dataset)
        print("Data load and preprocess done: {:.4f}s".format(perf_counter()-start_time))

    def define_model(self, hidden_dim, step_num, sample_num, nonlinear, agg):
        self.model = get_auto_model(self.feats.size(1), self.graph,
                            hidden_dim, step_num, sample_num, nonlinear, agg,
                            self.label_max, self.args.dropout)

    def delete_model(self):
        if self.model != None:
            del self.model

    def calc_loss(self, time, acc):
        if self.args.constraint:
            time = self.args.time_constraint - time
            if time > 0:
                accT = acc + dim_coff*log(time)
                return accT
            else:
                return -100
        else:
            acc = acc - self.args.accuracy_constraint
            if acc > 0:
                accT = -time + dim_coff*log(acc)
                return accT
            else:
                return -100

    def calc_f1(self, y_true, y_pred):
        if self.args.sigmoid:
            y_pred = F.sigmoid(y_pred)
            y_pred = y_pred.detach().numpy()
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0
        else:
            y_pred = torch.argmax(y_pred, dim=1)
        return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

    def train_model(self):
        train_idx = self.idx["train"]
        val_idx = self.idx["val"]
        ml = list()
        for index, module in enumerate(self.model.W):
            if (index == 0):
                ml.append({'params': module.parameters(), 'weight_decay':self.args.weight_decay})
            else:
                ml.append({'params': module.parameters()})
        ml.append({'params': self.model.lastW.parameters()})
        ml.append({'params': self.model.precomputeW.parameters()})
        optimizer = optim.Adam(ml, lr=self.args.lr)

        patient = 0
        loss = np.inf
        start_time = perf_counter()
        self.model.precomputeX(self.feats)
        for epoch in range(self.args.epochs):
            self.model.train()
            optimizer.zero_grad()
            output = self.model(self.feats)
            if self.args.sigmoid:
                loss_train = F.binary_cross_entropy(F.sigmoid(output[train_idx]), self.labels[train_idx])
            else:
                loss_train = F.cross_entropy(output[train_idx], self.labels[train_idx])
            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                self.model.eval()
                output = self.model(self.feats)
                if self.args.sigmoid:
                    new_loss = F.binary_cross_entropy(F.sigmoid(output[val_idx]), self.labels[val_idx])
                else:
                    new_loss = F.cross_entropy(output[val_idx], self.labels[val_idx])
                acc_mic, acc_mac = self.calc_f1(self.labels[val_idx], output[val_idx])
                if new_loss >= loss:
                    patient = patient + 1
                else:
                    patient = 0
                    loss = new_loss

            if patient == self.args.early_stopping:
               break

        train_time = perf_counter()-start_time
        return train_time

    def test_model(self):
        start_time = perf_counter()
        test_idx = self.idx["test"]
        self.model.eval()
        output = self.model(self.feats)
        acc_mic, acc_mac = self.calc_f1(self.labels[test_idx], output[test_idx])
        test_time = perf_counter()-start_time
        return acc_mic, acc_mac, test_time

