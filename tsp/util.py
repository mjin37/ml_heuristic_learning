import pickle
import elkai
import math
import torch
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from .solver import Solver
from torch.utils.data import DataLoader
from typing import Optional

MODEL_SAVE_FOLDER = './ckpt/'

# TODO: add Held-Karp lower bound calculation?

def save(model: object, args: dict, name: str) -> None:
    path = MODEL_SAVE_FOLDER + name
    torch.save(model.state_dict(), path)
    pickle.dump(args, open(path + '_args.pickle', 'wb'))

def load(model: object, name: str, device: str) -> object:
    path = MODEL_SAVE_FOLDER + name
    args = pickle.load(open(path + '_args.pickle', 'rb'))
    args.device = device
    model = Solver(
        model,
        args.embedding_size,
        args.hidden_size,
        args.seq_len,
        2,
        10,
        args.force_prob,
        args.heuristic
    )
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.to(device)
    model.eval()
    return model, args

CONST = 100000.0
def calc_dist(p, q):
    return np.sqrt(((p[1] - q[1])**2)+((p[0] - q[0])**2)) * CONST

def batch_calc_dist(p, q):
    return torch.sqrt(((p[:, 1] - q[:, 1])**2)+((p[:, 0] - q[:, 0])**2)) * CONST

def get_distance_matrix(pointset, device='cpu'):
    batch_size, num_points, dims = pointset.shape
    ret_matrix = torch.zeros((batch_size, num_points, num_points)).to(device)
    for i in range(num_points):
        for j in range(i+1, num_points):
            ret_matrix[:,i,j] = ret_matrix[:,j,i] = batch_calc_dist(pointset[:, i], pointset[:, j])
    return ret_matrix

def get_ref_reward(pointset):
    if isinstance(pointset, torch.cuda.FloatTensor):
        pointset = pointset.cpu()
    if isinstance(pointset, torch.FloatTensor):
        pointset = pointset.detach().numpy()

    num_points = len(pointset)
    ret_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i+1, num_points):
            ret_matrix[i,j] = ret_matrix[j,i] = calc_dist(pointset[i], pointset[j])
    q = elkai.solve_float_matrix(np.round(ret_matrix).astype(int)) # Output: [0, 2, 1]
    dist = 0
    for i in range(num_points):
        dist += ret_matrix[q[i], q[(i+1) % num_points]]
    return dist / CONST

def compute_solution(model, tsp_dataset):
    data_loader = DataLoader(
        tsp_dataset,
        batch_size=1
    )

    paths = [None] * len(data_loader)
    points = [None] * len(data_loader)
    for i, (_, tensor) in enumerate(data_loader):
        paths[i] = model(tensor)[2].detach().numpy()[0]
        points[i] = tensor.squeeze().detach().numpy()

    return paths[0]
