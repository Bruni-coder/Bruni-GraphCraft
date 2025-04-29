from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='D:/BruniResearch/GNN_Lab/data/Cora', name='Cora')
data = dataset[0]

print(f'节点数（num_nodes）: {data.num_nodes}')
print(f'边数（num_edges）: {data.num_edges}')
print(f'特征维度（num_node_features）: {data.num_node_features}')
print(f'类别数（num_classes）: {dataset.num_classes}')
print(f'是否是有向图（is_directed）: {data.is_directed()}')

print(data.x.shape)          # 节点特征矩阵形状
print(data.edge_index.shape) # 边的索引矩阵
print(data.y.shape)          # 标签向量形状
print(data.train_mask.shape) # 训练掩码形状
print(data.val_mask.shape)   # 验证掩码形状
print(data.test_mask.shape)  # 测试掩码形状

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

G = to_networkx(data, to_undirected=True)
plt.figure(figsize=(8, 8))
nx.draw(G, node_size=30, node_color=data.y, cmap=plt.cm.Set3, with_labels=False)
plt.title("Cora Graph - Node Colors = Labels")
plt.show()

