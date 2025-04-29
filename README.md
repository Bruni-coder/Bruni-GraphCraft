# Bruni's GNN Learning Lab

This repository documents my journey into Graph Neural Networks (GNNs) and PyTorch Geometric (PyG), from basic neural networks to constructing and training graph models.

---

## Project Structure

- `basic_gcn/`: Linear regression, MLP, function fitting experiments
- `dataset_exploration/`: Visual and statistical analysis of Cora dataset
- `advanced_gcn/`: GCN model construction and node classification
- `pytorch_basic/`: Hands-on PyTorch Sequential experiments
- `system_tools/`: Auxiliary system tests (optional)

---

## Highlights

- Built synthetic graphs manually using PyG
- Trained GCN models on custom and Cora datasets
- Explored different activation functions and network depths
- Visualized loss curves and function approximations
- Developed strong intuition for data structure and graph learning

---
## Structures
```
Bruni-GNN-Lab/
├── README.md                   # Project overview
├── requirements.txt            # Environment dependencies
├── .gitignore                  # Git ignore rules
│
├── basic_gcn/                  # Fundamental neural network exercises
│   ├── BruniAssignment1.py     # Linear regression
│   ├── BruniAssignment2.py     # Single hidden layer MLP
│   ├── BruniAssignment3.py     # Deep MLP and activation comparison
│   ├── BruniAssignment4.py     # Function fitting (sin/oscillations)
│
├── dataset_exploration/        # Dataset statistics and visualization
│   └── CorastatisticsTest.py   # Cora dataset analysis
│
├── advanced_gcn/               # Higher-level GCN experiments
│   ├── BruniAssignment5.py     # Manual PyG Data object creation
│   ├── BruniAssignment6.py     # Two-layer GCN for node classification
│   ├── GNN_Test.py             # Manual Cora loading + GCN training
│
├── pytorch_basic/              # Basic PyTorch syntax and model tests
│   └── Neural_network_demo.py  # Sequential model usage
│
└── system_tools/ (optional)    # System utilities
    └── Test_load.py            # Network connection testing script
```





## Requirements

- `torch`
- `torch-geometric`
- `networkx`
- `matplotlib`
- `numpy`
- (see `requirements.txt` for full list)

---

## License

This project is licensed under the MIT License.

---
*Authored by Bruni, 2025.*
