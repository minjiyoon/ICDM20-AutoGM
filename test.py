import torch
import torch.nn.functional as F
import torch.optim as optim

from args import get_args
from models import get_model, get_auto_model
from utils import load_citation, load_inductive_dataset

import numpy as np
from time import perf_counter
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

args = get_args()
args.inductive = args.dataset in ['reddit', 'ppi']
args.sigmoid = args.dataset in ['ppi']

def calc_f1(y_true, y_pred):
    if args.sigmoid:
        y_pred = F.sigmoid(y_pred)
        y_pred = y_pred.detach().numpy()
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    else:
        y_pred = torch.argmax(y_pred, dim=1)
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

def train_model(model, feats, labels, train_idx, val_idx,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, early_stop=args.early_stopping):

    ml = list()
    for index, module in enumerate(model.W):
        if (index == 0):
            ml.append({'params': module.parameters(), 'weight_decay':weight_decay})
        else:
            ml.append({'params': module.parameters()})
    ml.append({'params': model.lastW.parameters()})
    ml.append({'params': model.precomputeW.parameters()})
    optimizer = optim.Adam(ml, lr=lr)

    patient = 0
    loss = np.inf
    start_time = perf_counter()
    model.precomputeX(feats)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(feats)
        if args.sigmoid:
            loss_train = F.binary_cross_entropy(F.sigmoid(output[train_idx]), labels[train_idx])
        else:
            loss_train = F.cross_entropy(output[train_idx], labels[train_idx])
        loss_train.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            output = model(feats)
            if args.sigmoid:
                new_loss = F.binary_cross_entropy(F.sigmoid(output[val_idx]), labels[val_idx])
            else:
                new_loss = F.cross_entropy(output[val_idx], labels[val_idx])
            acc_mic, acc_mac = calc_f1(labels[val_idx], output[val_idx])
            if new_loss >= loss:
                patient = patient + 1
            else:
                patient = 0
                loss = new_loss
        if patient == early_stop:
            break

    train_time = perf_counter()-start_time
    return model, train_time

def test_model(model, feats, labels, test_idx):
    start_time = perf_counter()
    model.eval()
    output = model(feats)
    acc_mic, acc_mac = calc_f1(labels[test_idx], output[test_idx])
    test_time = perf_counter()-start_time
    return acc_mic, acc_mac, test_time

start_time = perf_counter()
if not args.inductive:
    graph, feats, labels, label_max, idx, _ = load_citation(args.dataset)
else:
    graph, graph_test, feats, labels, label_max, idx = load_inductive_dataset(args.dataset)

trial=10
acc_acc = 0
acc_train_time = 0
acc_test_time = 0
for _ in range(trial):
    model = get_auto_model(feats.size(1), graph,
                        args.hidden_dim, args.step_num, args.sample_num, args.nonlinear, args.aggregator,
                        label_max, args.dropout)

    model, train_time = train_model(model, feats, labels,
                                    idx["train"], idx["val"],
                                    args.epochs, args.weight_decay, args.lr, args.early_stopping)
    acc_train_time = acc_train_time + train_time

    acc_mic, acc_mac, test_time = test_model(model, feats, labels, idx["test"])
    acc_acc = acc_acc + acc_mic
    acc_test_time = acc_test_time + test_time

acc_acc = acc_acc/trial
acc_train_time = acc_train_time/trial
acc_test_time = acc_test_time/trial

print("{:.4f}\t{:.4f}\t{:.4f}".format(acc_train_time, acc_test_time, acc_acc))

