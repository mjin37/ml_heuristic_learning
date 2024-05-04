import pickle
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from .solver import solver_RNN
from torch.utils.data import DataLoader
from typing import Optional

MODEL_SAVE_FOLDER = './ckpt/'

def save(model: object, args: dict, name: str) -> None:
    path = MODEL_SAVE_FOLDER + name
    torch.save(model.state_dict(), path)
    pickle.dump(args, open(path + '_args.pickle', 'wb'))

def load(model: object, name: str, device: str) -> object:
    path = MODEL_SAVE_FOLDER + name
    args = pickle.load(open(path + '_args.pickle', 'rb'))
    args.device = device
    model = solver_RNN(
        model,
        args.embedding_size,
        args.hidden_size,
        args.seq_len,
        2,
        10
    )
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.to(device)
    model.eval()
    return model, args

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

def plot_tsp(points, path: Optional[torch.tensor] = None) -> None:

    """
    path: List of lists with the different orders in which the nodes are visited
    points: coordinates for the different nodes
    num_iters: number of paths that are in the path list
    
    """

    # Unpack the primary TSP path and transform it into a list of ordered 
    # coordinates

    if path is not None:
        ordered_points = points[path]
    else:
        ordered_points = points

    x = list(ordered_points[:, 0].numpy())
    y = list(ordered_points[:, 1].numpy())
    
    margin = 4 # buffer to add to the range
    y_min = min(y) - margin
    y_max = max(y) + margin
    x_min = min(x) - margin
    x_max = max(x) + margin

    # create map using BASEMAP
    map = Basemap(
        llcrnrlon=x_min,
        llcrnrlat=y_min,
        urcrnrlon=x_max,
        urcrnrlat=y_max,
        lat_0=(y_max - y_min)/2,
        lon_0=(x_max-x_min)/2,
        projection='merc',
        resolution = 'l',
        area_thresh=10000.
    )
    map.drawcoastlines()
    map.drawcountries()
    map.drawstates()
    map.drawmapboundary(fill_color='#46bcec')
    map.fillcontinents(color = 'white',lake_color='#46bcec')
    # convert lat and lon to map projection coordinates
    x, y = map(x, y)
    
    # plot points as red dots
    map.scatter(x, y, marker = 'o', color='b', zorder=5, s=10)

    a_scale = float(max(x))/float(60)

    if path is not None:
        # Draw the primary path for the TSP problem
        plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width = a_scale, 
                color ='blue', length_includes_head=True, zorder=5)
        for i in range(0,len(x)-1):
            plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width = a_scale,
                    color = 'blue', length_includes_head = True, zorder=5)
    plt.show()
