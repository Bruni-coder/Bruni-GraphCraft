import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(42)

x = torch.tensor([[1.0],[2.0],[3.0],[4.0],[5.0],[6.0],[7.0],[8.0],[9.0],[10.0],[11.0],[12.0],[13.0]])
y = torch.tensor([[1.0],[3.0],[27.0],[64.0],[125.0],[216],[343.0],[512.0],[729.0],[1000.0],[1331.0],[1728.0],[2197.0]])

class MyMoudle1(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1,16)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(16,8)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(8,1)

    def forward(self,x):
        x = self.hidden1(x)
        x = self.relu1(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x

model1 = MyMoudle1()

criterion = nn.MSELoss()
optimizer = optim.Adam(model1.parameters(),lr=0.005)

losses_relu = []

for epoch in range(10000):
    optimizer.zero_grad()
    y_pred = model1(x)
    loss = criterion(y_pred,y)
    losses_relu.append(loss.item())
    loss.backward()
    optimizer.step()

class MyMoudle2(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1,16)
        self.Tanh1 = nn.Tanh()
        self.hidden2 = nn.Linear(16,8)
        self.Tanh2 = nn.Tanh()
        self.output = nn.Linear(8,1)

    def forward(self,x):
        x = self.hidden1(x)
        x = self.Tanh1(x)
        x = self.hidden2(x)
        x = self.Tanh2(x)
        x = self.output(x)
        return x
model2 = MyMoudle2()

criterion = nn.MSELoss()
optimizer = optim.Adam(model2.parameters(),lr=0.005)

losses_tanh = []

for epoch in range(10000):
    optimizer.zero_grad()
    y_pred = model2(x)
    loss = criterion(y_pred,y)
    losses_tanh.append(loss.item())
    loss.backward()
    optimizer.step()

import matplotlib.pyplot as plt

x_n_smooth_np = x.detach().numpy()
y_n_smooth_np = y.detach().numpy()
y_pred1_n_smooth_np = model1(x).detach().numpy()
y_pred2_n_smooth_np = model2(x).detach().numpy()

# 构造更平滑的输入（100个点）
x_smooth = torch.linspace(1, 13, 100).reshape(-1, 1)

# 用两个模型分别预测
y_smooth1 = model1(x_smooth).detach().numpy()
y_smooth2 = model2(x_smooth).detach().numpy()

# 转成 NumPy，方便画图
x_smooth_np = x_smooth.detach().numpy()


plt.figure(figsize=(10,6))
plt.subplot(1,3,2)
plt.scatter(x_n_smooth_np,y_n_smooth_np,label='True Data',color='black',marker='o')
plt.plot(x_n_smooth_np,y_pred1_n_smooth_np,label='ReLU Model',color='red')
plt.plot(x_n_smooth_np,y_pred2_n_smooth_np,label='Tanh Model',color='blue')
plt.title('Model Comparison')
plt.legend()

plt.subplot(1,2,1)
plt.plot(losses_relu,label='relu loss',color='red')
plt.plot(losses_tanh,label='tanh loss',color='blue')
plt.title('Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 构造目标 y 值
y_true_smooth = x_smooth_np ** 3

# Error Comparison 图
plt.subplot(1,3,3)
plt.scatter(x_smooth_np, y_true_smooth - y_smooth1, label='RELU error', color='green')
plt.scatter(x_smooth_np, y_true_smooth - y_smooth2, label='TANH error', color='purple')
plt.title('Error Comparison')
plt.xlabel('x')
plt.ylabel('y_true - y_pred')
plt.legend()

plt.show()