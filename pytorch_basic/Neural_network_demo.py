import torch
import torch.nn as nn
x = torch.tensor([[2.0,4.0,7.0]])
model = nn.Sequential(
    nn.Linear(3,2),
    nn.Linear(2,1),
    nn.ReLU()
)
output = model(x)
print(output)