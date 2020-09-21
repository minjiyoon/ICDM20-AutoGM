import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Number of epochs to wait before early stop.')

    parser.add_argument('--dataset', type=str, default="cora",
                        help='Dataset to use.')
    parser.add_argument('--inductive', type=bool, default=False,
                        help='Inductive task (reddit, ppi).')
    parser.add_argument('--sigmoid', type=bool, default=False,
                        help='Multi-output task (ppi).')
    parser.add_argument('--decision_layer', type=str, default="classification",
                        choices=["classification","regression"],
                        help='Decision layer.')

    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--sample_num', type=int, default=-1,
                        help='Number of sampled neighbors')
    parser.add_argument('--step_num', type=int, default=2,
                        help='Number of propagating steps')
    parser.add_argument('--nonlinear', dest='nonlinear', action='store_true')
    parser.add_argument('--linear', dest='nonlinear', action='store_false')
    parser.set_defaults(nonlinear=True)
    parser.add_argument('--aggregator', type=int, default=4,
                        choices=[0,1,2,3,4,5],
                        help='Aggregation strategy')

    parser.add_argument('--time', dest='constraint', action='store_true')
    parser.add_argument('--accuracy', dest='constraint', action='store_false')
    parser.set_defaults(constraint=True)
    parser.add_argument('--time_constraint', type=float, default=100,
                        help='Upper bound of time')
    parser.add_argument('--accuracy_constraint', type=float, default=0.6,
                        help='Lower bound of accuracy')

    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
