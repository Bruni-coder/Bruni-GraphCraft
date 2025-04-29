import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

#对于正弦函数
x = torch.linspace(-10,10,100).reshape(-1,1)
y = torch.sin(x)
#正弦函数ReLU
class SinReLUmoudle(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1,128)
        self.relu = nn.ReLU()
        self.hidden2 = nn.Linear(128,64)
        self.relu = nn.ReLU()
        self.hidden3 = nn.Linear(64,10)
        self.relu = nn.ReLU()
        self.output = nn.Linear(10,1)

    def forward(self,x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden3(x)
        x = self.relu(x)
        x = self.output(x)
        return x
SinReLUmoudle = SinReLUmoudle()

criterion = nn.MSELoss()
optimizer = optim.Adam(SinReLUmoudle.parameters(),lr=0.001)

losses_relu = []

for epoch in range(10000):
    optimizer.zero_grad()
    y_pred = SinReLUmoudle(x)
    loss = criterion(y_pred,y)
    losses_relu.append(loss.item())
    loss.backward()
    optimizer.step()

#正弦函数Tanh
class SinTanhmoudle(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1,128)
        self.Tanh = nn.Tanh()
        self.hidden2 = nn.Linear(128,64)
        self.Tanh = nn.Tanh()
        self.hidden3 = nn.Linear(64,10)
        self.Tanh = nn.Tanh()
        self.output = nn.Linear(10,1)
    def forward(self,x):
        x = self.hidden1(x)
        x = self.Tanh(x)
        x = self.hidden2(x)
        x = self.Tanh(x)
        x = self.hidden3(x)
        x = self.Tanh(x)
        x = self.output(x)
        return x
SinTanhmoudle = SinTanhmoudle()

criterion = nn.MSELoss()
optimizer = optim.Adam(SinTanhmoudle.parameters(),lr=0.001)

losses_tanh = []

for epoch in range(10000):
    optimizer.zero_grad()
    y_pred = SinTanhmoudle(x)
    loss = criterion(y_pred,y)
    losses_tanh.append(loss.item())
    loss.backward()
    optimizer.step()

#正弦函数Sigmoid
class SinSigmoudle(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1,128)
        self.Sigmoid = nn.Sigmoid()
        self.hidden2 = nn.Linear(128,64)
        self.Sigmoid = nn.Sigmoid()
        self.hidden3 = nn.Linear(64,10)
        self.Sigmoid = nn.Sigmoid()
        self.output = nn.Linear(10,1)
    def forward(self,x):
        x = self.hidden1(x)
        x = self.Sigmoid(x)
        x = self.hidden2(x)
        x = self.Sigmoid(x)
        x = self.hidden3(x)
        x = self.Sigmoid(x)
        x = self.output(x)
        return x
SinSigmoudle = SinSigmoudle()

criterion = nn.MSELoss()
optimizer = optim.Adam(SinSigmoudle.parameters(),lr=0.001)

losses_sigmoid = []

for epoch in range(10000):
    optimizer.zero_grad()
    y_pred = SinSigmoudle(x)
    loss = criterion(y_pred,y)
    losses_sigmoid.append(loss.item())
    loss.backward()
    optimizer.step()

#对于震荡函数
x = torch.linspace(-10,10,100).reshape(-1,1)
y = x * torch.sin(x)
#震荡函数ReLU
class ShockReLUmoudle(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1,128)
        self.relu = nn.ReLU()
        self.hidden2 = nn.Linear(128,64)
        self.relu = nn.ReLU()
        self.hidden3 = nn.Linear(64,10)
        self.relu = nn.ReLU()
        self.output = nn.Linear(10,1)
    def forward(self,x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden3(x)
        x = self.relu(x)
        x = self.output(x)
        return x
ShockReLUmoudle = ShockReLUmoudle()

criterion = nn.MSELoss()
optimizer = optim.Adam(ShockReLUmoudle.parameters(),lr=0.001)

losses_relu = []

for epoch in range(10000):
    optimizer.zero_grad()
    y_pred = ShockReLUmoudle(x)
    loss = criterion(y_pred,y)
    losses_relu.append(loss.item())
    loss.backward()
    optimizer.step()

#震荡函数Tanh
class ShockTanhmoudle(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1,128)
        self.tanh = nn.Tanh()
        self.hidden2 = nn.Linear(128,64)
        self.tanh = nn.Tanh()
        self.hidden3 = nn.Linear(64,10)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(10,1)
    def forward(self,x):
        x = self.hidden1(x)
        x = self.tanh(x)
        x = self.hidden2(x)
        x = self.tanh(x)
        x = self.hidden3(x)
        x = self.tanh(x)
        x = self.output(x)
        return x
ShockTanhmoudle = ShockTanhmoudle()

criterion = nn.MSELoss()
optimizer = optim.Adam(ShockTanhmoudle.parameters(),lr=0.001)

losses_tanh = []

for epoch in range(10000):
    optimizer.zero_grad()
    y_pred = ShockTanhmoudle(x)
    loss = criterion(y_pred,y)
    losses_tanh.append(loss.item())
    loss.backward()
    optimizer.step()

#震荡函数Sigmoid
class ShockSigmoudle(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1,128)
        self.sigmoid = nn.Sigmoid()
        self.hidden2 = nn.Linear(128,64)
        self.sigmoid = nn.Sigmoid()
        self.hidden3 = nn.Linear(64,10)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(10,1)
    def forward(self,x):
        x = self.hidden1(x)
        x = self.sigmoid(x)
        x = self.hidden2(x)
        x = self.sigmoid(x)
        x = self.hidden3(x)
        x = self.sigmoid(x)
        x = self.output(x)
        return x
ShockSigmoudle = ShockSigmoudle()

criterion = nn.MSELoss()
optimizer = optim.Adam(ShockSigmoudle.parameters(),lr=0.001)

losses_sigmoid = []

for epoch in range(10000):
    optimizer.zero_grad()
    y_pred = ShockSigmoudle(x)
    loss = criterion(y_pred,y)
    losses_sigmoid.append(loss.item())
    loss.backward()
    optimizer.step()

#可视化
import matplotlib.pyplot as plt
x_n_smooth_np = x.detach().numpy()
y_n_smooth_np = y.detach().numpy()
y_pred1_n_smooth_np = SinReLUmoudle(x).detach().numpy()
y_pred2_n_smooth_np = SinTanhmoudle(x).detach().numpy()
y_pred3_n_smooth_np = SinSigmoudle(x).detach().numpy()
y_pred4_n_smooth_np = ShockReLUmoudle(x).detach().numpy()
y_pred5_n_smooth_np = ShockTanhmoudle(x).detach().numpy()
y_pred6_n_smooth_np = ShockSigmoudle(x).detach().numpy()

plt.figure(figsize=(10,6))
plt.subplot(2,3,1)
plt.scatter(x_n_smooth_np,y_n_smooth_np,label='True Data',color='black',marker='o')
plt.plot(x_n_smooth_np,y_pred1_n_smooth_np,label='ReLU Model',color='red')
plt.plot(x_n_smooth_np,y_pred2_n_smooth_np,label='Tanh Model',color='blue')
plt.plot(x_n_smooth_np,y_pred3_n_smooth_np,label='Sigmoid Model',color='green')
plt.title('Model Comparison')
plt.legend()

plt.subplot(2,3,2)
plt.plot(losses_relu,label='relu loss',color='red')
plt.plot(losses_tanh,label='tanh loss',color='blue')
plt.plot(losses_sigmoid,label='sigmoid loss',color='green')
plt.title('Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2,3,3)
plt.scatter(x_n_smooth_np,y_n_smooth_np-y_pred1_n_smooth_np,label='ReLU error',color='red')
plt.scatter(x_n_smooth_np,y_n_smooth_np-y_pred2_n_smooth_np,label='Tanh error',color='blue')
plt.scatter(x_n_smooth_np,y_n_smooth_np-y_pred3_n_smooth_np,label='Sigmoid error',color='green')
plt.title('Error Comparison')
plt.xlabel('x')
plt.ylabel('y_true - y_pred')
plt.legend()

plt.show()



