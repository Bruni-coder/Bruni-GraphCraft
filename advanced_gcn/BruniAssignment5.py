import torch
from torch_geometric.data import data

edge_index = torch.tensor([[0, 1, 1, 2, 0],
                           [1, 0, 2, 3, 0]], dtype=torch.long)

x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=torch.float)

data = data.Data(x=x, edge_index=edge_index)

print(data)