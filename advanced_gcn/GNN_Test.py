import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx

# 设置路径
prefix = 'D:/BruniResearch/GNN_Lab/data/Cora/ind.cora'

# 加载 pkl 文件（兼容 Python 2 和 3）
def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f, encoding='latin1')

# 读取文件
x = load_pickle(f'{prefix}.x')           # 训练节点特征
tx = load_pickle(f'{prefix}.tx')         # 测试节点特征
allx = load_pickle(f'{prefix}.allx')     # 所有训练节点特征
y = load_pickle(f'{prefix}.y')           # 训练标签
ty = load_pickle(f'{prefix}.ty')         # 测试标签
ally = load_pickle(f'{prefix}.ally')     # 所有训练标签
graph = load_pickle(f'{prefix}.graph')   # 邻接字典
test_idx = np.loadtxt(f'{prefix}.test.index', dtype=np.int32)
from scipy.sparse import vstack
# full_idx 用于重排特征矩阵，使训练集 + 测试集顺序一致
full_idx = list(range(allx.shape[0])) + test_idx.tolist()

# 合并训练 + 测试特征矩阵
features = vstack([allx, tx]).tolil()
features = features[full_idx].toarray()

# 合并标签
labels = np.vstack((ally, ty))[full_idx]

# 重排测试索引（test.index 是乱序）
full_idx = list(range(allx.shape[0])) + test_idx.tolist()
features = features[full_idx]
labels = labels[full_idx]

# 转换为 PyTorch 格式
x = torch.tensor(features, dtype=torch.float)
y = torch.tensor(np.argmax(labels, axis=1), dtype=torch.long)

# 构建边列表
G = nx.from_dict_of_lists(graph)
edge_index = torch.tensor(list(G.edges)).t().contiguous()

# 创建训练和测试 mask
train_mask = torch.zeros(y.size(0), dtype=torch.bool)
train_mask[:140] = True
test_mask = torch.zeros(y.size(0), dtype=torch.bool)
test_mask[-1000:] = True  # 最后 1000 个为测试

# 构造 Data 对象
data = Data(x=x, edge_index=edge_index, y=y,
            train_mask=train_mask, test_mask=test_mask)
# 定义 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data.x.size(1), 16)
        self.conv2 = GCNConv(16, y.max().item() + 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, p=0.5, train=self.training)
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)

# 训练
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = torch.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 测试
model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)
correct = pred[data.test_mask] == data.y[data.test_mask]
acc = int(correct.sum()) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
