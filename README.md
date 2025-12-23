# GNN配体受体预测模型

基于图神经网络的配体受体相互作用预测深度学习模型，专为基因表达数据设计。

## 🎯 项目概述

- **目标**: 基于基因表达数据预测配体受体对分数
- **数据规模**: 468,260样本 × 5,051基因 → 1,120配体受体对
- **核心技术**: 层次化图注意力网络 (H-GAT)
- **特色功能**: 自动GPU资源管理、混合精度训练、超参数优化

## 📁 项目结构

```
xenium_gnn/
├── config.yaml                 # 主配置文件
├── requirements.txt            # 依赖包列表
├── setup_env.sh               # 环境设置脚本
├── train.py                   # 主训练脚本
├── README.md                  # 项目说明
├── src/                       # 源代码
│   ├── data/
│   │   └── preprocessing.py   # 数据预处理
│   ├── models/
│   │   └── gnn_model.py      # GNN模型架构
│   ├── training/
│   │   ├── trainer.py        # 训练器
│   │   └── hyperopt.py       # 超参数优化
│   ├── evaluation/
│   │   └── evaluator.py      # 模型评估
│   └── utils/
│       └── gpu_utils.py      # GPU工具
├── data/                      # 数据文件
│   ├── x.csv                 # 基因表达矩阵
│   ├── y.csv                 # 配体受体分数
│   └── pathway_gene.csv      # 通路基因映射
├── models/                    # 保存的模型
├── logs/                      # 训练日志
└── results/                   # 结果输出
```

## 🚀 快速开始

### 1. 环境准备

确保您在conda的deeplearning环境中：

```bash
# 激活环境并检查
source setup_env.sh
```

### 2. 安装依赖

```bash
# 在deeplearning环境中安装
conda activate deeplearning
pip install -r requirements.txt
```

### 3. 数据准备

确保数据文件位于正确位置：
- `../data/x.csv`: 基因表达矩阵
- `../data/y.csv`: 配体受体分数
- `../data/pathway_gene.csv`: 通路基因映射

### 4. 模型训练

```bash
# 基础训练
python train.py --mode train

# 超参数优化
python train.py --mode hyperopt

# 模型评估
python train.py --mode evaluate
```

## ⚙️ 配置说明

主要配置项在`config.yaml`中：

### 模型配置
- `gene_embedding_dim`: 基因嵌入维度 (默认: 128)
- `hidden_dim`: 隐藏层维度 (默认: 256)
- `gat_layers`: GAT层数 (默认: 3)
- `attention_heads`: 注意力头数 (默认: 8)

### 训练配置
- `batch_size`: 批次大小 (默认: 64)
- `learning_rate`: 学习率 (默认: 0.001)
- `epochs`: 训练轮数 (默认: 100)
- `patience`: 早停耐心值 (默认: 15)

### GPU配置
- `use_gpu`: 是否使用GPU (默认: true)
- `mixed_precision`: 混合精度训练 (默认: true)
- `device_ids`: GPU设备ID列表 (默认: [0])

## 🔧 核心功能

### 1. 自动GPU资源管理
- 自动检测可用GPU
- 智能选择最佳GPU
- 动态内存管理
- 混合精度训练支持

### 2. 层次化图注意力网络
- 基因嵌入层
- 多层GAT处理
- 通路级特征聚合
- 残差连接和层归一化

### 3. 智能训练策略
- 自适应学习率调度
- 早停机制
- 梯度裁剪
- 检查点保存

### 4. 超参数优化
- Optuna框架
- TPE采样器
- 中位数剪枝
- 可视化结果

### 5. 全面评估
- 多种回归指标
- 按目标变量评估
- 预测结果可视化
- 残差分析

## 📊 模型架构

```python
GNNLigandReceptorModel(
  (gene_embedding): Embedding(5051, 128)
  (input_projection): Linear(1, 128)
  (gat): HierarchicalGAT(
    (gat_layers): ModuleList(
      (0-2): 3 x GATConv(128, 128, heads=8)
    )
    (layer_norms): ModuleList(
      (0-2): 3 x LayerNorm
    )
  )
  (pathway_aggregator): PathwayAggregator(128, 256)
  (predictor): Sequential(
    (0): Linear(256, 512)
    (1): ReLU()
    (2): Dropout(0.1)
    (3): Linear(512, 256)
    (4): ReLU()
    (5): Dropout(0.1)
    (6): Linear(256, 1120)
  )
)
```

## 📝 更新日志

### v1.0.0 (2025-12)
- 初始版本发布
- 基础GNN模型实现
- 自动GPU管理
- 超参数优化支持
- 完整评估体系

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见LICENSE文件
