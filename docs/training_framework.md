# GNN配体受体预测模型训练框架与流程详解

本文件系统性、完整地叙述 `\home\lilj\work\xenium\code` 项目中配体受体预测模型的训练框架与端到端流程，包括数据入口、配置、模型结构、训练策略、评估、交叉验证与超参优化、GPU管理与日志输出、命令行接口，以及常见问题与最佳实践。

## 总览

- 入口脚本：`train.py`，通过参数选择训练、评估、交叉验证、超参优化四种模式，支持配置覆盖与测试数据快速切换。
- 数据管线：`src/data/preprocessing.py` 提供全流程数据处理，产出 `DataLoader` 与 `metadata`。
- 模型架构：`src/models/gnn_model.py` 实现层次化 GAT 与通路聚合器，输出多目标预测分数。
- 训练器：`src/training/trainer.py` 管理优化器、损失、学习率调度、早停、混合精度、可视化与检查点。
- 评估器：`src/evaluation/evaluator.py` 提供回归与分类指标、图表与报告生成。
- 交叉验证与超参优化：`src/training/cross_validation.py`、`src/training/hyperopt.py`。
- GPU管理：`src/utils/gpu_utils.py` 自动检测、选择设备并监控内存；支持混合精度。

## 项目结构与配置

- 关键文件
  - `train.py`
  - `config.yaml`
  - `src/data/preprocessing.py`
  - `src/models/gnn_model.py`
  - `src/training/trainer.py`
  - `src/evaluation/evaluator.py`
  - `src/training/cross_validation.py`
  - `src/training/hyperopt.py`
  - `src/utils/gpu_utils.py`

- 配置文件：`config.yaml`
  - 数据源、划分与归一化：`data.*`
  - 模型结构：`model.*`
  - 训练策略：`training.*`
  - GPU与并行：`gpu.*`
  - 优化器与调度：`optimizer.*`、`scheduler.*`
  - 评估指标：`evaluation.metrics`
  - 输出路径：`output.*`
  - 任务类型与分类阈值：`task_type`、`classification.*`

示例配置参考：`config.yaml:9-40,44-68,72-93,97-112,116-128,132-143,147-154,157-170,174-189,193-203,206-209`。

## 命令行入口与模式

- 入口与参数解析：`train.py:41-75`
  - `--mode {train, hyperopt, evaluate, cv}` 切换工作模式
  - `--config` 指定配置文件
  - `--checkpoint` 恢复训练
  - `--log-level` 日志级别
  - `--conda-env` 环境提示
  - 数据覆盖与 H5 支持：`--classification`、`--data-dir`、`--file-format {csv,h5}`、`--x-key`、`--y-key`、`--padj-key`
  - 交叉验证：`--cv-type {kfold,stratified}`、`--cv-folds`、`--save-cv-models`
  - 测试数据切换：`--test`（自动选择 `testdata/cell` 优先）

- 模式分支：
  - 训练模式：`train.py:209-233,243-267`
  - 评估模式：`train.py:268-316`
  - 超参优化：`train.py:167-183`
  - 交叉验证：`train.py:184-208`

## 数据处理管线（DataPreprocessor）

核心流程：`src/data/preprocessing.py:408-458`

1) 加载数据：`load_data()`（`preprocessing.py:86-216`）
   - 自动选择 `csv` 或 `h5`，支持自定义 H5 键名与列名读取（`preprocessing.py:139-163,171-195`）
   - 分类任务优先使用 `padj`，并在缺失时回退（含任务类型动态切换）：`preprocessing.py:101-125`
   - 路径与文件严格校验：`preprocessing.py:131-138,169-176,206-216`

2) 过滤无信息标签（分类）：`_filter_all_zero_lr_pairs()`（`preprocessing.py:70-84`）

3) 构建映射：
   - 基因到索引：`build_gene_mapping()`（`preprocessing.py:218-224`）
   - 通路到基因：`build_pathway_mapping()`（`preprocessing.py:225-243`）

4) 构建图结构（通路内基因完全图，双向）：`build_graph()`（`preprocessing.py:244-272`）

5) 标准化与标签处理：`normalize_data()`（`preprocessing.py:274-321`）
   - 基因表达标准化（`StandardScaler`）
   - 分类标签按阈值生成 0/1：`preprocessing.py:200-204`
   - 回归标签按 `lr_normalize_method` 选择：`log1p`/`standard`/`minmax`/`none`（`preprocessing.py:288-307`）

6) 数据集划分（train/val/test）：`split_data()`（`preprocessing.py:323-362`）

7) 构建 `DataLoader`（共享图与通路映射，`custom_collate_fn`）：`create_dataloaders()`（`preprocessing.py:364-407`，`preprocessing.py:19-31`）

8) 返回元数据：`metadata` 包含维度与映射、边索引等（`preprocessing.py:445-458`）。

数据形状约定：
- `gene_expr`：`[batch_size, num_genes]`
- `lr_scores`：`[batch_size, num_lr_pairs]`
- `edge_index`：`[2, num_edges]`（所有样本共享）
- `pathway_mapping`：`Dict[str, List[int]]`

## 模型架构（GNNLigandReceptorModel）

实现文件：`src/models/gnn_model.py`

- 基因嵌入与输入投影：`gnn_model.py:204-209,276-287`
  - `Embedding(num_genes, gene_embedding_dim)`
  - 将表达值 `1->gene_embedding_dim` 线性映射并与嵌入相加

- 图神经网络（层次化 GAT）：`HierarchicalGAT`（`gnn_model.py:110-185`）
  - 多层 `GATConv`，残差与层归一化（`gnn_model.py:142-155,167-184`）
  - 处理批次时合并节点与边索引：`gnn_model.py:288-318,320-324`

- 通路级聚合：`PathwayAggregator`（`gnn_model.py:14-45,46-108`）
  - 支持 `attention`/`transformer`/`mean`/`max` 等聚合策略
  - 将每条通路的基因嵌入聚合，再对通路集合做均值后映射到 `hidden_dim`

- 预测头：多层感知机输出所有配体受体对分数（`gnn_model.py:226-235`）

- 前向流程：`forward()`（`gnn_model.py:250-331`）
  - 输入字典包含 `gene_expr`、`edge_index`、`pathway_mapping`
  - 批量图构建 → GAT → 通路聚合 → 预测分数

## 训练流程（GNNTrainer）

实现文件：`src/training/trainer.py`

- 初始化：`trainer.py:102-165`
  - 设备选择与混合精度：`trainer.py:109-123`（GPU 由 `GPUManager` 管理：`gpu_utils.py:71-101,164-171`）
  - 优化器：Adam/AdamW/SGD（`trainer.py:166-196`）
  - 损失：分类 `BCEWithLogitsLoss`，回归 `MSELoss`（`trainer.py:127-131`）
  - 学习率调度：`cosine`/`step`/`plateau` + `warmup`（`trainer.py:133-139,57-98`）
  - 早停：耐心值与最小改善（`trainer.py:141-146,24-55`）

- 单轮训练：`train_epoch()`（`trainer.py:207-268`）
  - 支持 AMP（`autocast` + `GradScaler`），可选梯度裁剪（`trainer.py:231-236,248-252`）
  - 按需清理 GPU 缓存（`trainer.py:265-267`）

- 验证：`validate_epoch()`（`trainer.py:270-305`）
  - 分类将 `logits -> sigmoid` 后评估 AUC/F1 等（`trainer.py:290-297`）
  - 指标由 `ModelEvaluator` 计算（`evaluator.py:21-48,96-121`）

- 训练主循环：`train()`（`trainer.py:375-490`）
  - 学习率记录、GPU内存监控、指标曲线与可视化（`trainer.py:410-473,497-371`）
  - 检查点保存与最佳模型（`trainer.py:306-332,476-480`）

输出物：
- 模型与检查点：`output.model_dir`（例如 `./models`）
- 日志与图表：`output.log_dir`、`output.result_dir`（例如 `./logs`、`./results`）
- 训练历史图：`training_history.png`（`trainer.py:333-373`）

## 评估与报告（ModelEvaluator）

实现文件：`src/evaluation/evaluator.py`

- 指标集合：`mse`、`mae`、`rmse`、`r2`、`pearson`、`spearman`、`explained_variance` 以及分类指标 `accuracy_micro/macro`、`f1_micro/macro`、`roc_auc_micro/macro`、`precision_micro/macro`、`recall_micro/macro`（`evaluator.py:21-42`）。
- 评估入口：`evaluate()`（`evaluator.py:96-121`）、逐目标评估：`evaluate_per_target()`（`evaluator.py:123-158`）。
- 图表：预测散点、残差分布与 Q-Q 图（英文标题，便于国际化）（`evaluator.py:160-237,238-290`）。
- 报告生成与保存：`generate_report()`（数值 JSON、CSV 与图表）（`evaluator.py:291-367`）。
- 分类指标细节：在样本全零时自动跳过以避免偏置（`evaluator.py:373-405,454-505,506-521`）。

## 交叉验证（CrossValidator）

实现文件：`src/training/cross_validation.py`

- 支持标准 K 折与分层 K 折（回归分层通过分箱）：`cross_validation.py:33-58,59-95`。
- 主流程：`run_cross_validation()` 构造每折的 `DataLoader`、训练与评估，并汇总均值与标准差（`cross_validation.py:97-230`）。
- 结果保存与图表：JSON、文本摘要、折线图与箱线图（`cross_validation.py:270-402`）。
- 配置注意：该模块使用 `self.config['paths']['models/results']` 保存文件（`cross_validation.py:184-195,270-324`）。当前 `config.yaml` 使用 `output.*`，如需在交叉验证中保存模型与结果，请在配置中补充 `paths.models` 与 `paths.results` 或调整模块以复用 `output.*`。

## 超参数优化（HyperparameterOptimizer）

实现文件：`src/training/hyperopt.py`

- 搜索空间与试验：`hyperopt.search_space`、`n_trials`、`timeout`（`config.yaml:174-189`，`hyperopt.py:38-50,182-226`）。
- 目标函数：对建议参数刷新配置，训练至较短轮次并返回最佳验证损失（`hyperopt.py:109-175`）。
- 结果保存与可视化：最佳参数 JSON、试验历史 CSV/PNG/HTML、参数重要性与关系图（`hyperopt.py:235-414`）。

## GPU 管理与混合精度

实现文件：`src/utils/gpu_utils.py`

- 自动检测与选择设备，记录总/已用/可用内存（`gpu_utils.py:19-52,54-70,71-101`）。
- 运行时内存监控与缓存清理（`gpu_utils.py:103-119`）。
- 混合精度启用条件与 `GradScaler`（`gpu_utils.py:164-171`）。

## 训练与评估示例命令

- 基础训练（CSV）
  - `python train.py --mode train`

- 使用 H5 数据与键名（分类任务）
  - `python train.py --classification --file-format h5 --data-dir /home/lilj/work/xenium/data/cell --x-key X --padj-key padj --mode train`

- 使用测试数据（优先 `testdata/cell`）：`train.py:129-155`
  - `python train.py --test --mode train`

- 交叉验证（5 折，保存每折模型）
  - `python train.py --mode cv --cv-type kfold --cv-folds 5 --save-cv-models`

- 模型评估（加载最佳模型并在测试集上报告）
  - `python train.py --mode evaluate`

- 超参优化（Optuna）
  - `python train.py --mode hyperopt`

## 输出目录结构

- `output.model_dir`: 保存检查点与 `best_model.pth`（`trainer.py:306-332`）
- `output.log_dir`: 训练日志与可视化（如每 epoch 可视化）
- `output.result_dir`: 指标曲线、评估图与报告（`trainer.py:333-373,497-371`，`train.py:257-267`）
- 交叉验证输出见 `cross_validation.py:270-402`（注意 `paths.*` 与 `output.*` 的差异）

## 任务类型与标签约定

- 任务类型来源于配置或命令行覆盖：`train.py:110-116,118-128`；数据侧动态切换见 `preprocessing.py:114-125`。
- 分类任务：
  - `padj_threshold` 生成二分类标签（`config.yaml:206-209`，`preprocessing.py:200-204`）
  - 训练时用 `BCEWithLogitsLoss` 与概率评估（`trainer.py:127-131,290-297`）
  - 预测阈值 `pred_threshold` 可用于推理阶段（`trainer.py:106-108`）
- 回归任务：
  - 标签归一化通过 `lr_normalize_method` 控制（`config.yaml:34-40`，`preprocessing.py:288-307`）

## 日志与错误处理

- 全局日志初始化与文件输出：`train.py:24-33`（`training.log`）
- 关键阶段日志：配置加载、GPU 检测、数据规模、训练/评估指标与时间统计（`train.py:93-101,159-166,228-233,234-267`）。
- 错误捕获与堆栈输出：`train.py:322-326`。

## 常见问题与最佳实践

- CUDA 内存不足
  - 减小 `training.batch_size`；启用 `gpu.mixed_precision`；必要时关闭可视化或提升 `num_workers` 以平衡 IO。
  - 参考 `gpu_utils.get_memory_usage()` 在训练循环中记录（`trainer.py:415-419`）。

- 指标为零或异常
  - 分类任务存在全零样本/目标时，评估器会自动跳过，确保指标不失真（`evaluator.py:373-405,454-505`）。
  - 检查 `padj_threshold` 是否过严，或数据质量（`preprocessing.py:70-84,200-204`）。

- 交叉验证结果保存路径不一致
  - 当前 `cross_validation.py` 使用 `config['paths']`，而主配置使用 `output.*`。需要在配置中新增：
    - `paths.models: ./models`
    - `paths.results: ./results/cross_validation`
  - 或调整模块统一走 `output.*`。

- H5 列名缺失
  - 读取失败时自动回退到 `gene_0..N` 与 `lr_0..N`，但建议确保 H5 包含列名或辅助数据集（`preprocessing.py:144-163,178-195`）。

## 代码位置速览（关键引用）

- 参数解析与模式选择：`train.py:41-75`
- 测试数据覆盖逻辑：`train.py:129-155`
- 模型创建：`src/models/gnn_model.py:369-381`
- 训练入口：`src/training/trainer.py:375-490`
- 检查点与最佳模型：`src/training/trainer.py:306-332,476-480`
- 评估报告生成：`src/evaluation/evaluator.py:291-367`
- 交叉验证主流程：`src/training/cross_validation.py:97-230`
- 超参优化主流程：`src/training/hyperopt.py:182-226`
- GPU 管理与混合精度：`src/utils/gpu_utils.py:71-101,164-171`

## 结语

该训练框架面向大规模基因表达与配体受体预测任务，强调端到端的可靠性与可视化可解释性。通过统一的数据接口、可配置的模型与训练策略、完备的评估与报告，以及可选的交叉验证与超参数优化，能够在不同数据形态（CSV/H5，回归/分类）与硬件条件下稳定运行。