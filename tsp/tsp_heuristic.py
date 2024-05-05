import elkai
import numpy as np
import torch

CONST = 100000.0
def calc_dist(p, q):
    return np.sqrt(((p[1] - q[1])**2)+((p[0] - q[0])**2)) * CONST

def get_ref_reward(pointset):
    if isinstance(pointset, torch.cuda.FloatTensor) or isinstance(pointset, torch.FloatTensor):
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


def nearest_neighbor(distance_matrix,sub_tour):
    """This function returns the next node from the nearest neighbor sequential heuristic.
    Input: distance_matrix: 2D tensor whose i,j^th element is the distance on i -> j
    sub_tour: list of selected city indices
    """
    if len(sub_tour) == 0:
        return 0
    last_city = sub_tour[-1]
    num_cities = distance_matrix.size()[0]
    curr_min_dist = np.inf
    chosen_city_index = 0
    for i in range(num_cities):
        if i in sub_tour:
            continue
        else:
            curr_dist = distance_matrix[last_city,i]
            if curr_dist < curr_min_dist:
                curr_min_dist = curr_dist
                chosen_city_index = i

    return chosen_city_index
    


#distance_matrix =torch.rand(5,5)
#sub_tour = [0,2,3]

distance_matrix = torch.zeros((5,5))+1
distance_matrix[3,4] = 0
distance_matrix[4,3] = 0
sub_tour = [1,4]
print(nearest_neighbor(distance_matrix,sub_tour))

