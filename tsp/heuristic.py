import elkai
import numpy as np
import torch

CONST = 100000.0
def calc_dist(p, q):
    return np.sqrt(((p[1] - q[1])**2)+((p[0] - q[0])**2)) * CONST

def batch_calc_dist(p, q):
    return torch.sqrt(((p[:, 1] - q[:, 1])**2)+((p[:, 0] - q[:, 0])**2)) * CONST

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

def get_distance_matrix(pointset, device='cpu'):
    batch_size, num_points, dims = pointset.shape
    ret_matrix = torch.zeros((batch_size, num_points, num_points)).to(device)
    for i in range(num_points):
        for j in range(i+1, num_points):
            ret_matrix[:,i,j] = ret_matrix[:,j,i] = batch_calc_dist(pointset[:, i], pointset[:, j])
    return ret_matrix


# def get_ordered_sub_tour(sub_tour,placements):
#     len_sub_tour = len(sub_tour)
#     ordered_sub_tour = np.zeros(len_sub_tour,dtype = int)

#     return ordered_sub_tour

def nearest_neighbor(pointset, subtour, placements=None, device='cpu'):
    """
    This function returns the next node from the nearest neighbor sequential heuristic.
    
    Args:
        distance_matrix: 2D matrix whose i,j^th element is the distance on i -> j
        subtour: list of selected city indices, placements is not used for this
    """
    distance_matrix = get_distance_matrix(pointset, device=device)
    batch_size, num_cities, _ = distance_matrix.shape
    if subtour.size == 0:
        raise ValueError("Subtour is empty.")
    last_city = subtour[:, -1][:, None]
    curr_min_dist = 2 * CONST * torch.ones(batch_size).to(device)
    chosen_city_index = torch.zeros(batch_size).to(device)
    for i in range(num_cities):
        curr_dist = distance_matrix[:, :, i].gather(dim=1, index=last_city).squeeze(1)
        curr_dist += torch.nan_to_num(np.inf * torch.sum(i == subtour, dim=1))
        chosen_city_index[curr_dist < curr_min_dist] = i
        curr_min_dist, _ = torch.min(torch.stack((curr_dist, curr_min_dist)), dim=0)

    return chosen_city_index
    
# def nearest(distance_matrix,sub_tour,placement):
#     """Input: distance_matrix: 2D tensor, distance from i -> j is i,j^th element,
#     sub_tour: list of city indices,
#     placement: list of index of placement of each city"""

#     if len(sub_tour) == 0:
#         return 0
#     num_cities = np.shape(distance_matrix)[0]
#     #we find the closest city to the subtour
#     min_dist = np.inf
#     best_city = 0
#     for i in range(num_cities):
#         if i in sub_tour:
#             continue
#         local_min_dist = np.inf
#         for j in sub_tour:
#             curr_dist = distance_matrix[i,j]
#             if curr_dist < local_min_dist:
#                 local_min_dist = curr_dist
#         if local_min_dist < min_dist:
#             min_dist = local_min_dist
#             best_city = i
#     #now we find the optimal place to put the city


    
    
            
        



# def farthest(distance_matrix,sub_tour,placement):
#     """Input: distance_matrix: 2D tensor, distance from i -> j is i,j^th element,
#     sub_tour: list of city indices,
#     placement: list of index of placement of each city"""
#     if len(sub_tour) == 0:
#         return 0
#     num_cities = np.shape(distance_matrix)[0]
#     #we find the farthest city to the subtour
#     for i in range(num_cities):
#         if i in sub_tour:
#             continue
#         max_dist = -np.inf
#         best_city = 0
#         for i in range(num_cities):
#             if i in sub_tour:
#                 continue
#             local_max_dist = -np.inf
#             for j in sub_tour:
#                 curr_dist = distance_matrix[i,j]
#                 if curr_dist > local_max_dist:
#                     local_max_dist = curr_dist
#             if local_max_dist > max_dist:
#                 max_dist = local_max_dist
#                 best_city = i
# print(get_distance_matrix(torch.tensor([[1,2],[3,4]])))
