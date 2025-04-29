import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN,self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)

    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

edge_index = torch.tensor([[0,0,0,0,1,1,1,2,2,3,3,4,4],
                           [1,2,3,4,0,2,4,1,3,2,4,2,1]], dtype=torch.long)
x = torch.tensor([[1.0,0.0,0.0,1.0],
                  [1.0,1.0,0.0,1.0],
                  [1.0,1.0,1.0,1.0],
                  [1.0,0.0,0.0,0.0],
                  [0.0,0.0,0.0,1.0]],dtype=torch.float)
y = torch.tensor([0.1,0.2,0.3,0.4,0.5],dtype=torch.long)
data = Data(x=x,edge_index=edge_index)

model = GCN(in_channels=4, out_channels=5)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.5)

for epoch in range(10000):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, y)
    loss.backward() 
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

model.eval()
out = model(data)
_, pred = out.max(dim=1)
print("Predicted labels:", pred)

