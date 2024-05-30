import torch
from tsp.heuristic import InsertionHeuristic

pointset = torch.tensor([[1,1],[2,1],[0,0],[2,2],[10,10],[-10,-10]])
subtour = torch.tensor([2,0,3])

pointset2 = torch.tensor([[0,0],[5,5],[10,10],[1,1],[8,8],[9.9,9.8]])
subtour2 = torch.tensor([0,4,2])

pointset = torch.stack((pointset,pointset2))
subtour = torch.stack((subtour,subtour2))

print(pointset)
print(subtour)
print()

heuristic = InsertionHeuristic("nearest")
print(heuristic.get_next_index(pointset, subtour))
