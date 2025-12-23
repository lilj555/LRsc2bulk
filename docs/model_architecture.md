# GNN配体受体预测模型架构

## 概述

本模型采用创新的层次化图注意力网络（Hierarchical Graph Attention Network, H-GAT）架构，专门设计用于从单细胞基因表达数据中预测配体受体（Ligand-Receptor, LR）相互作用的强度或显著性。该架构融合了深度学习、图神经网络和生物信息学的最新进展，能够同时处理回归（预测相互作用强度）和分类（预测相互作用显著性）两种任务类型。

### 核心设计理念

1. **层次化建模**: 模型采用基因→通路→配体受体的层次化信息处理流程，模拟生物系统中信息传递的自然层次结构。

2. **注意力机制**: 在整个架构中广泛使用注意力机制，包括基因级别的图注意力和通路级别的聚合注意力，确保模型能够关注最相关的特征。

3. **多任务兼容**: 统一的架构设计使得同一模型可以无缝切换回归和分类任务，只需调整输出层和损失函数。

4. **生物学可解释性**: 通过基因嵌入和通路聚合，模型不仅提供预测结果，还能揭示哪些基因和通路对特定配体受体相互作用贡献最大。

## 通用架构组件

### 1. 基因嵌入层 (`src/models/gnn_model.py:205`)
```python
self.gene_embedding = nn.Embedding(self.num_genes, self.gene_embedding_dim)
```

**详细功能描述**:
基因嵌入层是模型的基础组件，它为每个基因学习一个高维向量表示。这些嵌入向量捕获了基因在配体受体相互作用预测任务中的语义信息。与传统的one-hot编码不同，嵌入层能够学习基因之间的相似性关系，使得功能相似的基因在嵌入空间中距离更近。

**技术细节**:
- **输入**: 基因索引（整数），范围从0到`num_genes-1`
- **输出**: 形状为`[num_genes, gene_embedding_dim]`的嵌入矩阵
- **初始化**: 使用Xavier均匀初始化，确保训练稳定性
- **可学习参数**: 每个基因都有一个独立的嵌入向量，总共`num_genes × gene_embedding_dim`个参数

**生物学意义**:
基因嵌入可以理解为将每个基因映射到一个连续向量空间中，其中向量的方向和大小编码了该基因在细胞通讯网络中的功能角色。经过训练后，这些嵌入可以用于基因功能相似性分析、通路富集分析等下游任务。

### 2. 输入投影层 (`src/models/gnn_model.py:208`)
```python
self.input_projection = nn.Linear(1, self.gene_embedding_dim)
```

**详细功能描述**:
输入投影层负责将原始的基因表达值（通常是TPM、FPKM或UMI计数）映射到与基因嵌入相同的维度空间。这一设计允许模型同时利用基因的身份信息（通过嵌入层）和表达量信息（通过投影层），实现信息的有效融合。

**技术细节**:
- **输入**: 基因表达值，形状为`[batch_size, num_genes, 1]`
- **输出**: 形状为`[batch_size, num_genes, gene_embedding_dim]`的投影表达特征
- **操作**: 对每个基因的表达值独立进行线性变换
- **融合方式**: 投影后的表达特征与基因嵌入相加，实现信息融合

**设计原理**:
这种设计允许模型区分"哪些基因被表达"（通过嵌入层）和"表达了多少"（通过投影层）。相加操作（而不是拼接）减少了参数数量，同时保持了信息的可加性，这在生物学上是合理的——基因的功能角色和表达水平共同决定了其生物学效应。

### 3. 层次化GAT (`src/models/gnn_model.py:211-217`)
```python
self.gat = HierarchicalGAT(
    input_dim=self.gene_embedding_dim,
    hidden_dim=self.gene_embedding_dim,
    num_layers=self.gat_layers,
    num_heads=self.attention_heads,
    dropout=self.dropout
)
```

**详细功能描述**:
层次化图注意力网络是模型的核心组件，它在基因相互作用图上进行信息传播和特征学习。GAT通过多头注意力机制，让每个基因节点能够自适应地关注其邻居节点中最重要的信息，从而捕获复杂的基因调控关系。

**架构细节**:
- **图结构**: 基于基因相互作用网络构建，边表示已知的蛋白质-蛋白质相互作用、共表达关系或通路共成员关系
- **注意力机制**: 每个注意力头学习不同的相互作用模式，多头设计增强了模型的表达能力
- **层次化**: 多层GAT允许信息在多跳邻居间传播，捕获更广泛的网络拓扑信息
- **残差连接**: 在深层GAT中可能包含残差连接，缓解梯度消失问题

**生物学解释**:
GAT层模拟了细胞内信号传导的过程：每个基因（节点）从其相互作用的伙伴（邻居节点）接收信息，并通过注意力权重决定哪些信息更重要。这类似于生物系统中分子通过选择性结合和信号放大来实现特异性响应。

### 4. 通路聚合器 (`src/models/gnn_model.py:220-224`)
```python
self.pathway_aggregator = PathwayAggregator(
    input_dim=self.gene_embedding_dim,
    output_dim=self.hidden_dim,
    aggregation_type=config['model']['pathway_aggregation']
)
```

**详细功能描述**:
通路聚合器将基因级别的特征提升到通路级别的表示。它接收经过GAT处理的所有基因特征，按照预定义的通路-基因映射关系，将属于同一通路的基因特征聚合起来，生成通路级别的特征表示。

**聚合策略**:
1. **Attention聚合**: 使用多头注意力机制，让模型学习通路中不同基因的重要性权重
2. **Transformer聚合**: 使用Transformer编码器，捕获基因间的复杂依赖关系
3. **Mean聚合**: 简单平均，假设通路中所有基因贡献相等
4. **Max聚合**: 取最大值，关注通路中最活跃的基因

**技术优势**:
- **灵活性**: 支持多种聚合策略，可根据具体任务选择最合适的方法
- **可解释性**: Attention权重可以揭示通路中哪些基因对预测结果最重要
- **维度统一**: 无论通路包含多少基因，输出都是固定维度的向量

**生物学意义**:
通路级别的聚合反映了生物学中的一个重要概念——通路（Pathway）作为功能单元。许多细胞过程不是由单个基因完成，而是由一组协同工作的基因共同实现。通路聚合使得模型能够从系统生物学角度理解细胞通讯。

## 回归模型架构

### 预测头设计原理 (`src/models/gnn_model.py:227-235`)
```python
self.predictor = nn.Sequential(
    nn.Linear(self.hidden_dim, self.hidden_dim * 2),
    nn.ReLU(),
    nn.Dropout(self.dropout),
    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
    nn.ReLU(),
    nn.Dropout(self.dropout),
    nn.Linear(self.hidden_dim, self.num_lr_pairs)
)
```

**详细架构分析**:
回归任务的预测头采用三层全连接网络结构，具有以下设计特点：

1. **特征扩展**: 第一层将隐藏维度从`hidden_dim`扩展到`hidden_dim * 2`，增加模型的表达能力，允许捕获更复杂的特征交互。

2. **非线性变换**: 使用ReLU激活函数引入非线性，使模型能够学习复杂的映射关系。ReLU的选择基于其计算效率和避免梯度消失问题的优势。

3. **正则化机制**: 每层后都添加Dropout，防止过拟合。Dropout率与模型其他部分保持一致（默认0.1），确保正则化的一致性。

4. **维度压缩**: 最终层将特征压缩到`num_lr_pairs`维度，直接输出每个配体受体对的预测强度分数。

**输出解释**:
回归任务的输出是连续的实数值，表示预测的配体受体相互作用强度。这些分数可以直接解释为相互作用的相对强度或亲和力。

### 损失函数设计与优化 (`src/training/trainer.py:130`)
```python
self.criterion = nn.MSELoss()
```

**损失函数选择原理**:
均方误差（MSE）损失被选择用于回归任务，原因如下：

1. **数学性质**: MSE是连续变量预测中最常用的损失函数，它对大误差给予更高的惩罚，促使模型更关注准确预测。

2. **可微性**: MSE处处可微，便于梯度下降优化。

3. **生物学合理性**: 在配体受体相互作用预测中，强度的相对大小比绝对精度更重要，MSE能够很好地反映这种需求。

**梯度特性**:
MSE损失的梯度与误差成正比，这意味着：
- 大误差产生大梯度，加速模型修正
- 小误差产生小梯度，避免过度调整
- 这种特性特别适合生物学数据中常见的长尾分布

### 前向传播流程详解

#### 1. 基因嵌入生成 (`src/models/gnn_model.py:276-278`)
```python
gene_indices = torch.arange(num_genes, device=gene_expr.device)
gene_emb = self.gene_embedding(gene_indices)  # [num_genes, embedding_dim]
```

**技术细节**:
- 使用`torch.arange`生成所有基因的索引，确保批处理一致性
- 嵌入层将这些离散索引映射到连续的向量空间
- 嵌入在训练过程中不断优化，捕获基因功能语义

#### 2. 表达值融合 (`src/models/gnn_model.py:280-286`)
```python
gene_expr_expanded = gene_expr.unsqueeze(-1)  # [batch_size, num_genes, 1]
expr_emb = self.input_projection(gene_expr_expanded)  # [batch_size, num_genes, embedding_dim]
gene_emb_batch = gene_emb.unsqueeze(0).expand(batch_size, -1, -1)
combined_emb = gene_emb_batch + expr_emb  # [batch_size, num_genes, embedding_dim]
```

**融合策略分析**:
- **unsqueeze操作**: 为基因表达值添加最后一个维度，便于线性投影
- **广播机制**: 基因嵌入通过广播扩展到批次维度，与表达特征维度匹配
- **相加融合**: 使用元素级相加而不是拼接，减少参数数量同时保持信息完整性

#### 3. 图神经网络处理 (`src/models/gnn_model.py:288-323`)
**图构建过程**:
- 每个样本独立处理，但共享相同的图结构
- 边索引根据批次大小进行调整，确保正确的节点连接
- 使用基因掩码过滤只涉及基因节点的边

**信息传播**:
- GAT层让每个基因节点从其邻居聚合信息
- 多头注意力捕获不同类型的基因相互作用
- 多层设计允许信息在多跳邻居间传播

#### 4. 通路级聚合 (`src/models/gnn_model.py:325-326`)
```python
pathway_features = self.pathway_aggregator(gat_output, pathway_mapping)  # [batch_size, hidden_dim]
```

**聚合意义**:
- 将基因级别的特征提升到通路级别
- 减少特征维度，提高计算效率
- 增强模型对生物学功能单元的理解

#### 5. 回归预测 (`src/models/gnn_model.py:329`)
```python
predictions = self.predictor(pathway_features)  # [batch_size, num_lr_pairs]
```

**输出特性**:
- 输出形状与标签完全一致
- 每个值对应一个配体受体对的预测强度
- 数值范围没有限制，反映真实的相互作用强度分布

### 评估指标体系 (`src/evaluation/evaluator.py:24-30`)
```python
'mse': self._calculate_mse,        # 均方误差
'mae': self._calculate_mae,        # 平均绝对误差
'rmse': self._calculate_rmse,      # 均方根误差
'r2': self._calculate_r2,          # R²分数
'pearson': self._calculate_pearson, # Pearson相关系数
'spearman': self._calculate_spearman, # Spearman相关系数
'explained_variance': self._calculate_explained_variance # 解释方差
```

**评估指标详解**:

1. **MSE (均方误差)**: 衡量预测值与真实值之间的平方差异，对异常值敏感
2. **MAE (平均绝对误差)**: 更稳健的误差度量，不受异常值影响
3. **RMSE (均方根误差)**: 与原始数据单位一致的误差度量
4. **R² (决定系数)**: 表示模型解释的方差比例，范围[0,1]
5. **Pearson相关系数**: 衡量线性相关程度，范围[-1,1]
6. **Spearman相关系数**: 衡量单调相关程度，对非线性关系更稳健
7. **解释方差**: 模型捕获的目标变量方差比例

## 分类模型架构

### 预测头结构一致性
分类任务使用与回归完全相同的预测头结构，这种设计具有重要优势：

**设计 rationale**:
1. **架构统一**: 相同的网络结构允许模型权重在任务间迁移
2. **参数共享**: 预训练的回归模型可以微调用于分类任务
3. **计算效率**: 避免维护两套不同的预测头参数

**输出解释差异**:
虽然网络结构相同，但输出的解释完全不同：
- 回归: 输出直接作为相互作用强度分数
- 分类: 输出作为logits，需要经过sigmoid转换为概率

### 损失函数设计 (`src/training/trainer.py:128`)
```python
self.criterion = nn.BCEWithLogitsLoss()
```

**BCEWithLogitsLoss 优势**:
1. **数值稳定性**: 结合sigmoid和BCE损失，避免数值计算问题
2. **梯度特性**: 提供良好的梯度信号，便于优化
3. **概率校准**: 直接输出logits，便于后续的概率转换

**与回归损失对比**:
- 回归关注连续值的精确预测
- 分类关注类别边界的正确划分
- 这种差异反映了不同的生物学问题 formulation

### 标签生成流程 (`src/data/preprocessing.py:201-203`)
```python
if self.task_type == 'classification':
    logger.info(f"使用padj阈值 {self.padj_threshold} 生成二分类标签")
    labels_df = (labels_df <= self.padj_threshold).astype(int)
```

**统计学基础**:
- padj (调整后p值) 是多重假设检验校正后的显著性指标
- 阈值0.05是生物学中常用的显著性水平
- 二值化将连续统计量转换为离散类别标签

**标签分布考虑**:
- 正负样本比例影响模型训练
- 极端不平衡时需要采样策略或损失加权
- 阈值选择需要在灵敏性和特异性间权衡

### 概率转换机制 (`src/training/trainer.py:292`)
```python
probs = torch.sigmoid(predictions).cpu()
```

**sigmoid函数特性**:
- 将logits映射到(0,1)区间，解释为概率
- 梯度在接近0.5时最大，便于模型调整决策边界
- 输出可以解释为相互作用显著性的置信度

**决策阈值**:
- 默认使用0.5作为二分类阈值
- 可根据具体应用调整阈值平衡精确率和召回率
- 多标签分类中每个配体受体对独立判断

### 评估指标体系 (`src/evaluation/evaluator.py:32-41`)
```python
'accuracy_micro': self._accuracy_micro,      # 微平均准确率
'accuracy_macro': self._accuracy_macro,      # 宏平均准确率
'f1_micro': self._f1_micro,                  # 微平均F1分数
'f1_macro': self._f1_macro,                  # 宏平均F1分数
'roc_auc_micro': self._roc_auc_micro,        # 微平均ROC-AUC
'roc_auc_macro': self._roc_auc_macro,        # 宏平均ROC-AUC
'precision_micro': self._precision_micro,    # 微平均精确率
'precision_macro': self._precision_macro,    # 宏平均精确率
'recall_micro': self._recall_micro,          # 微平均召回率
'recall_macro': self._recall_macro           # 宏平均召回率
```

**分类评估详解**:

1. **微平均 vs 宏平均**:
   - 微平均: 所有样本平等加权，反映整体性能
   - 宏平均: 所有类别平等加权，反映类别平衡性能

2. **ROC-AUC**: 衡量模型在不同阈值下的整体分类能力
   - 值越接近1，模型区分能力越强
   - 对类别不平衡不敏感

3. **F1分数**: 精确率和召回率的调和平均
   - 适合不平衡数据集评估
   - 在生物学应用中通常比准确率更有意义

4. **多标签特性**:
   - 每个配体受体对作为一个独立的二分类任务
   - 评估指标聚合所有对的性能
   - 反映模型在多标签预测中的整体能力

## 模型参数统计

### 参数初始化 (`src/models/gnn_model.py:240-249`)
```python
def _initialize_weights(self):
    """初始化模型权重"""
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
```

### 默认配置参数
```yaml
model:
  gene_embedding_dim: 128    # 基因嵌入维度
  hidden_dim: 256           # 隐藏层维度
  gat_layers: 3             # GAT层数
  attention_heads: 8        # 注意力头数
  dropout: 0.1              # Dropout率
  pathway_aggregation: "attention"  # 通路聚合方式
```

## 任务类型切换

### 配置设置 (`config.yaml:206-209`)
```yaml
task_type: regression  # 可选：regression 或 classification
classification:
  padj_threshold: 0.05  # padj阈值，小于等于此值标记为阳性（1）
  pred_threshold: 0.5   # 推理时概率阈值，将概率转换为0/1
```

### 自动检测 (`src/data/preprocessing.py:119-122`)
```python
# 如果没有y.csv而存在padj.csv，则回退到分类
labels_path = padj_path
self.task_type = 'classification'
self.config['task_type'] = 'classification'
```

## 性能特点

1. **可扩展性**: 支持任意数量的基因和配体受体对
2. **灵活性**: 通过配置轻松切换回归/分类任务
3. **可解释性**: 提供基因嵌入和通路聚合的可视化
4. **高效性**: 利用GPU加速和混合精度训练

## 文件位置参考

- 主模型类: `src/models/gnn_model.py:187`
- 训练器: `src/training/trainer.py:106-130`
- 数据预处理: `src/data/preprocessing.py:65-68,201-203`
- 评估器: `src/evaluation/evaluator.py:24-42`
- 配置文件: `config.yaml:206-209`