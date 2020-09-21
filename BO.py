import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.optim as optim

from bayes_opt import BayesianOptimization
from time import perf_counter
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

from args import get_args
from models import get_model, get_auto_model
from utils import load_citation, load_inductive_dataset
from unifiedGM import UnifiedGM

trial = 10
UGM = UnifiedGM()

def rounds(hidden_dim, step_num, sample_num, nonlinear, aggregator):

    if hidden_dim > 256:
        hidden_dim = 256
    elif hidden_dim < 1:
        hidden_dim = 1
    else:
        hidden_dim = round(hidden_dim)

    if step_num > 30:
        step_num = 30
    elif step_num < 1:
        step_num = 1
    else:
        step_num = round(step_num)

    if sample_num > 30:
        sample_num = -1
    elif sample_num < 1:
        sample_num = 1
    else:
        sample_num = round(sample_num)

    if nonlinear > 0.5:
        nonlinear = True
    else:
        nonlinear = False

    if nonlinear > 0.5:
        nonlinear = True
    else:
        nonlinear = False

    aggregator = math.floor(aggregator)

    return int(hidden_dim), int(step_num), int(sample_num), nonlinear, aggregator

def black_box_function(hidden_dim, step_num, sample_num, nonlinear, aggregator):
    acc_acc = 0
    acc_train_time = 0
    acc_test_time = 0
    for _ in range(trial):
        UGM.define_model(*rounds(hidden_dim, step_num, sample_num, nonlinear, aggregator))
        train_time = UGM.train_model()
        acc_train_time = acc_train_time + train_time
        acc_mic, acc_mac, test_time = UGM.test_model()
        acc_acc = acc_acc + acc_mic
        acc_test_time = acc_test_time + test_time
        UGM.delete_model()
    acc_acc = acc_acc/trial
    acc_train_time = acc_train_time/trial
    acc_test_time = acc_test_time/trial
    print("Accuracy: {}, Training time: {}, Test time: {}\n".format(acc_acc, acc_train_time, acc_test_time))
    return UGM.calc_loss(acc_test_time, acc_acc)

# Bounded region of parameter space
pbounds = {'hidden_dim': (1, 256), 'step_num': (1, 5), 'sample_num': (1,60), 'nonlinear': (0, 1), 'aggregator': (0, 6)}

import random
import time
optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds,\
                                 random_state=random.seed(time.time()))

optimizer.maximize(init_points=30, n_iter=0)

print(optimizer.max)




