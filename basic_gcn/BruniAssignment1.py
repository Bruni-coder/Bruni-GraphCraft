import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

x = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y = torch.tensor([[4.0],[7.0],[10.0],[13.0]])
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        return self.linear(x)
model = LinearRegression()

print(model.linear.weight.data)
print(model.linear.bias.data)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)

for epoch in range(15):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred,y)
    loss.backward()
    optimizer.step()

print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
print("训练前参数：")
print("权重 =", model.linear.weight.data)
print("偏置 =", model.linear.bias.data)

# ...训练循环...

print("训练后参数：")
print("权重 =", model.linear.weight.data)
print("偏置 =", model.linear.bias.data)



