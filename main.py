#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print("PyTorch has version {}".format(torch.__version__))
# print('Using device:', device)

from tsp.util import save, load, compute_solution
from tsp.train import train
from tsp.data import TSPDataset
from tsp.heuristic import NearestNeighbor, NearestInsertion, FarthestInsertion
from tsp.model import RNNTSP

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

class TSPArgs(argparse.Namespace):
    def __init__(self):
        self.model = RNNTSP
        self.seq_len = 30
        self.num_epochs = 200
        self.num_tr_dataset = 10000
        self.num_te_dataset = 2000
        self.embedding_size = 128
        self.hidden_size = 128
        self.batch_size = 64
        self.grad_clip = 1.5
        self.use_cuda = True
        self.beta = 0.9
        self.force_prob = 0.0
        self.heuristic = NearestNeighbor(device)
        self.reg = 1.0
        self.name = f"reg=0.1-fp=0-0"
        self.disable_tqdm = None

args = TSPArgs()

train_dataset = TSPDataset()
test_dataset = TSPDataset()

seed = 0
train_dataset.random_fill(args.seq_len, args.num_tr_dataset, random_seed=seed)
test_dataset.random_fill(args.seq_len, args.num_te_dataset, random_seed=seed)

model = train(train_dataset, test_dataset, args)
save(model, args, args.name)

# for fp in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
#     args.force_prob = fp
#     args.name = f"force_prob={fp}-{seed}"
#     print(f"Force Probability = {args.force_prob}")
#     model = train(train_dataset, test_dataset, args)
#     save(model, args, args.name)
