import elkai
import numpy as np
import torch

"""
TSP heuristic types:
    1. Tour construction
      + a. Nearest Neighbor: Go to the the nearest city until no unvisited
        cities.
      + b. Greedy: Construct tour by adding shortest edge that doesn't create a
        cycle or increase the degree of any node.
      + c. Nearest/Farthest Insertion: Start with shortest edge as subtour,
        select a city that has the shortest/farthest distance to subtour, find
        edge s.t. insertion of city is minimum cost, repeat.
      / d. Convex Hull: Start with convex hull as subtour, find cheapest
        insertion for each city not in subtour, insert cheapest, repeat.
        (nearest insertion but initialize with convex hull)
      - e. Christofides: Start with MST, create minimum weight matching on odd
        degree nodes and add to MST, create Euler cycle from combined graph and
        traverse with shortcuts
    2. Tour improvement
        a. 2-opt: Given a tour, remove two edges and reconnect (cross) the two
        paths if cheaper, repeat until convergence (2-optimal)
        b. 3-opt: Given a tour, remove three edges and reconnect if cheaper,
        repeat until convergence
        c. k-opt: Generalization of the above
        d. Lin-Kernighan: Dynamically selects k at each iteration
        (see https://arxiv.org/pdf/2110.07983)
        e: Tabu-search/Simulated Annealing/Genetic Algorithms: Extensions of the
        above

ref: http://160592857366.free.fr/joe/ebooks/ShareData/Heuristics%20for%20the%20Traveling%20Salesman%20Problem%20By%20Christian%20Nillson.pdf
"""

class TSPHeuristic:
    def __init__(self, device):
        self.device = device

    def __call__(self, pointset, subtour):
        pass

class NearestNeighbor(TSPHeuristic):
    def __call__(self, pointset, subtour):
        eps = 1e-6
        B, V, D = pointset.shape
        S = subtour.shape[-1]
        subpoints = pointset.gather(1, subtour.unsqueeze(2).expand(-1, -1, D))
        last = subpoints[:, -1, :].unsqueeze(1)
        dists = torch.cdist(last, pointset)
        mask, i = torch.ones(B, V).to(self.device), torch.arange(B).long().to(self.device)
        mask[i[:, None], subtour] = np.inf
        dists = (dists + eps) * mask.unsqueeze(1)
        _, idx = torch.min(dists, dim=-1)

        return idx.squeeze(-1)


class InsertionHeuristic(TSPHeuristic):
    def __init__(self, mode, device):
        super(InsertionHeuristic, self).__init__(device)

        self.mode = mode
        
    def __call__(self, pointset, subtour):
        eps = 1e-6
        B, V, D = pointset.shape
        S = subtour.shape[-1]
        subpoints = pointset.gather(1, subtour.unsqueeze(2).expand(-1, -1, D))
        dists = torch.cdist(subpoints, pointset)
        mask, i = torch.ones(B, V).to(device), torch.arange(B).long().to(device)

        if self.mode == "nearest":
            mask[i[:, None], subtour] = np.inf
            dists = (dists + eps) * mask.unsqueeze(1).expand(B, S, V)
            n_min, n_idx = torch.min(dists, dim=-1)
            s_min, s_idx = torch.min(n_min, dim=-1)
        elif self.mode == "farthest":
            mask[i[:, None], subtour] = -np.inf
            dists = (dists + eps) * mask.unsqueeze(1).expand(B, S, V)
            n_min, n_idx = torch.max(dists, dim=-1)
            s_min, s_idx = torch.max(n_min, dim=-1)
        ins = n_idx.gather(1, s_idx.unsqueeze(1)).squeeze(-1)
        inspoints = pointset[i, ins]

        sub_dists = torch.norm(subpoints - torch.roll(subpoints, -1, dims=1), dim=-1)
        ins_dists = torch.cdist(subpoints, inspoints.unsqueeze(1)).squeeze(-1)
        nsi_dists = torch.roll(ins_dists, -1, dims=1)

        costs = ins_dists + nsi_dists - sub_dists
        idx = subtour[i, torch.argmin(costs, dim=-1)]

        return ins, idx


class NearestInsertion(InsertionHeuristic):
    def __init__(self, device):
        super(NearestInsertion, self).__init__("nearest", device)


class FarthestInsertion(InsertionHeuristic):
    def __init__(self, device):
        super(FarthestInsertion, self).__init__("farthest", device)
