#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("PyTorch has version {}".format(torch.__version__))
print('Using device:', device)

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
        self.num_epochs = 100
        self.num_tr_dataset = 10000
        self.num_te_dataset = 10 # 2000
        self.embedding_size = 128
        self.hidden_size = 128
        self.batch_size = 64
        self.grad_clip = 1.5
        self.use_cuda = False
        self.beta = 0.9
        self.force_prob = 1.0
        self.heuristic = NearestNeighbor()
        self.name = f"force_prob={self.force_prob}"
        self.disable_tqdm = None


args = TSPArgs()
train_dataset = TSPDataset()
test_dataset = TSPDataset()
train_dataset.random_fill(args.seq_len, args.num_tr_dataset)
test_dataset.random_fill(args.seq_len, args.num_te_dataset)

model = train(train_dataset, test_dataset, args)
save(model, args, args.name)
