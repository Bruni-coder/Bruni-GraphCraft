import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(42)

x = torch.tensor([[1.0],[2.0],[3.0],[4.0],[5.0]])
y = torch.tensor([[1.0],[2.0],[9.0],[16.0],[25.0]])

class MyMoudle(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1,10)
        self.relu = nn.ReLU()
        self.output = nn.Linear(10,1)

    def forward(self,x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x
model = MyMoudle()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)

for epoch in range(10000):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred,y)
    loss.backward()
    optimizer.step()

print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("训练前参数：")
print("权重 =", model.hidden.weight.data)
print("偏置 =", model.hidden.bias.data)

print("训练后参数：")
print("权重 =", model.output.weight.data)
print("偏置 =", model.output.bias.data)