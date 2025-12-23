"""
交叉验证模块
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Dict, List, Tuple, Optional, Generator
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.gnn_model import create_model
from ..training.trainer import GNNTrainer
from ..evaluation.evaluator import ModelEvaluator
from ..utils.gpu_utils import GPUManager
from ..data.preprocessing import custom_collate_fn

logger = logging.getLogger(__name__)

class CrossValidator:
    """交叉验证器"""
    
    def __init__(self, config: dict, metadata: dict):
        self.config = config
        self.metadata = metadata
        self.gpu_manager = GPUManager()
        self.results = []
        
    def k_fold_split(self, dataset, n_splits: int = 5, shuffle: bool = True, 
                     random_state: int = 42) -> Generator[Tuple[Subset, Subset], None, None]:
        """
        K折交叉验证数据分割
        
        Args:
            dataset: 数据集
            n_splits: 折数
            shuffle: 是否打乱
            random_state: 随机种子
            
        Yields:
            (train_subset, val_subset): 训练和验证子集
        """
        kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        indices = list(range(len(dataset)))
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            logger.info(f"Fold {fold + 1}/{n_splits}: 训练样本 {len(train_idx)}, 验证样本 {len(val_idx)}")
            
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            yield train_subset, val_subset
    
    def stratified_k_fold_split(self, dataset, targets: np.ndarray, n_splits: int = 5, 
                               shuffle: bool = True, random_state: int = 42) -> Generator[Tuple[Subset, Subset], None, None]:
        """
        分层K折交叉验证数据分割（用于分类任务）
        
        Args:
            dataset: 数据集
            targets: 目标标签
            n_splits: 折数
            shuffle: 是否打乱
            random_state: 随机种子
            
        Yields:
            (train_subset, val_subset): 训练和验证子集
        """
        # 对于回归任务，我们可以将连续值分箱来实现分层
        if len(targets.shape) > 1:
            # 多目标回归，使用第一个目标的分箱
            target_for_stratify = targets[:, 0]
        else:
            target_for_stratify = targets
        
        # 将连续值分为5个箱子用于分层
        n_bins = min(5, len(np.unique(target_for_stratify)))
        stratify_labels = pd.cut(target_for_stratify, bins=n_bins, labels=False)
        
        skfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        indices = list(range(len(dataset)))
        
        for fold, (train_idx, val_idx) in enumerate(skfold.split(indices, stratify_labels)):
            logger.info(f"Stratified Fold {fold + 1}/{n_splits}: 训练样本 {len(train_idx)}, 验证样本 {len(val_idx)}")
            
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            yield train_subset, val_subset
    
    def run_cross_validation(self, dataset, test_dataset=None, cv_type: str = "kfold", 
                           n_splits: int = 5, save_models: bool = False) -> Dict:
        """
        运行交叉验证
        
        Args:
            dataset: 训练数据集
            test_dataset: 测试数据集（可选）
            cv_type: 交叉验证类型 ("kfold" 或 "stratified")
            n_splits: 折数
            save_models: 是否保存每折的模型
            
        Returns:
            交叉验证结果字典
        """
        logger.info(f"开始 {n_splits} 折交叉验证 (类型: {cv_type})")
        
        # 获取目标值用于分层（如果需要）
        targets = None
        if cv_type == "stratified":
            targets = np.array([dataset[i]['lr_scores'].numpy() for i in range(len(dataset))])
        
        # 选择分割方法
        if cv_type == "stratified" and targets is not None:
            split_generator = self.stratified_k_fold_split(dataset, targets, n_splits)
        else:
            split_generator = self.k_fold_split(dataset, n_splits)
        
        fold_results = []
        
        for fold, (train_subset, val_subset) in enumerate(split_generator):
            logger.info(f"=" * 50)
            logger.info(f"开始第 {fold + 1} 折训练")
            logger.info(f"=" * 50)
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_subset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=self.config['gpu']['num_workers'],
                pin_memory=self.config['gpu']['pin_memory'],
                collate_fn=custom_collate_fn
            )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                num_workers=self.config['gpu']['num_workers'],
                pin_memory=self.config['gpu']['pin_memory'],
                collate_fn=custom_collate_fn
            )
            
            # 创建模型
            model = create_model(self.config, self.metadata)
            
            # 创建训练器
            trainer = GNNTrainer(
                model=model,
                config=self.config,
                metadata=self.metadata
            )
            
            # 训练模型
            fold_history = trainer.train(train_loader, val_loader)
            
            # 评估模型
            evaluator = ModelEvaluator(self.config['evaluation']['metrics'])
            
            # 验证集评估
            val_metrics = evaluator.evaluate_model(model, val_loader, trainer.device)
            
            # 测试集评估（如果提供）
            test_metrics = None
            if test_dataset is not None:
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.config['training']['batch_size'],
                    shuffle=False,
                    num_workers=self.config['gpu']['num_workers'],
                    pin_memory=self.config['gpu']['pin_memory'],
                    collate_fn=custom_collate_fn
                )
                test_metrics = evaluator.evaluate_model(model, test_loader, trainer.device)
            
            # 保存模型（如果需要）
            if save_models:
                model_path = Path(self.config['paths']['models']) / f"cv_fold_{fold + 1}_model.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': self.config,
                    'metadata': self.metadata,
                    'fold': fold + 1,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics
                }, model_path)
                logger.info(f"模型已保存: {model_path}")
            
            # 记录结果
            fold_result = {
                'fold': fold + 1,
                'train_history': fold_history,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'best_epoch': trainer.best_epoch,
                'best_val_loss': trainer.best_val_loss
            }
            
            fold_results.append(fold_result)
            
            logger.info(f"第 {fold + 1} 折完成:")
            logger.info(f"  最佳验证损失: {trainer.best_val_loss:.6f}")
            val_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            logger.info(f"  验证指标 - {val_str}")
            if test_metrics:
                test_str = ", ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()])
                logger.info(f"  测试指标 - {test_str}")
        
        # 汇总结果
        cv_results = self._summarize_cv_results(fold_results)
        
        # 保存结果
        self._save_cv_results(cv_results, cv_type, n_splits)
        
        # 生成可视化
        self._plot_cv_results(cv_results, cv_type, n_splits)
        
        logger.info("=" * 60)
        logger.info("交叉验证完成!")
        val_mean_str = ", ".join([f"{k}: {cv_results['val_metrics_mean'][k]:.4f} ± {cv_results['val_metrics_std'][k]:.4f}" for k in cv_results['val_metrics_mean'].keys()])
        logger.info(f"平均验证指标 - {val_mean_str}")
        if cv_results['test_metrics_mean']:
            test_mean_str = ", ".join([f"{k}: {cv_results['test_metrics_mean'][k]:.4f} ± {cv_results['test_metrics_std'][k]:.4f}" for k in cv_results['test_metrics_mean'].keys()])
            logger.info(f"平均测试指标 - {test_mean_str}")
        logger.info("=" * 60)
        
        return cv_results
    
    def _summarize_cv_results(self, fold_results: List[Dict]) -> Dict:
        """汇总交叉验证结果"""
        
        # 提取所有指标
        val_metrics_list = [result['val_metrics'] for result in fold_results]
        test_metrics_list = [result['test_metrics'] for result in fold_results if result['test_metrics']]
        
        # 计算验证指标的均值和标准差
        val_metrics_mean = {}
        val_metrics_std = {}
        
        if val_metrics_list:
            metric_names = val_metrics_list[0].keys()
            for metric in metric_names:
                values = [metrics[metric] for metrics in val_metrics_list]
                val_metrics_mean[metric] = np.mean(values)
                val_metrics_std[metric] = np.std(values)
        
        # 计算测试指标的均值和标准差
        test_metrics_mean = {}
        test_metrics_std = {}
        
        if test_metrics_list:
            metric_names = test_metrics_list[0].keys()
            for metric in metric_names:
                values = [metrics[metric] for metrics in test_metrics_list]
                test_metrics_mean[metric] = np.mean(values)
                test_metrics_std[metric] = np.std(values)
        
        return {
            'fold_results': fold_results,
            'val_metrics_mean': val_metrics_mean,
            'val_metrics_std': val_metrics_std,
            'test_metrics_mean': test_metrics_mean if test_metrics_list else None,
            'test_metrics_std': test_metrics_std if test_metrics_list else None,
            'n_folds': len(fold_results)
        }
    
    def _save_cv_results(self, cv_results: Dict, cv_type: str, n_splits: int):
        """保存交叉验证结果"""
        results_dir = Path(self.config['paths']['results'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        results_file = results_dir / f"cv_results_{cv_type}_{n_splits}fold.json"
        
        # 转换numpy类型为Python原生类型以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        cv_results_serializable = convert_numpy(cv_results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(cv_results_serializable, f, indent=2, ensure_ascii=False)
        
        logger.info(f"交叉验证结果已保存: {results_file}")
        
        # 保存汇总统计
        summary_file = results_dir / f"cv_summary_{cv_type}_{n_splits}fold.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"交叉验证汇总报告\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"验证类型: {cv_type}\n")
            f.write(f"折数: {n_splits}\n\n")
            
            f.write("验证集指标 (均值 ± 标准差):\n")
            for metric, mean_val in cv_results['val_metrics_mean'].items():
                std_val = cv_results['val_metrics_std'][metric]
                f.write(f"  {metric}: {mean_val:.6f} ± {std_val:.6f}\n")
            
            if cv_results['test_metrics_mean']:
                f.write("\n测试集指标 (均值 ± 标准差):\n")
                for metric, mean_val in cv_results['test_metrics_mean'].items():
                    std_val = cv_results['test_metrics_std'][metric]
                    f.write(f"  {metric}: {mean_val:.6f} ± {std_val:.6f}\n")
        
        logger.info(f"交叉验证汇总已保存: {summary_file}")
    
    def _plot_cv_results(self, cv_results: Dict, cv_type: str, n_splits: int):
        """绘制交叉验证结果图表"""
        results_dir = Path(self.config['paths']['results'])
        
        # 提取指标数据
        fold_numbers = [result['fold'] for result in cv_results['fold_results']]
        
        # 主要指标
        main_metrics = ['mse', 'mae', 'r2', 'pearson_corr']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(main_metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # 验证集指标
            val_values = [result['val_metrics'].get(metric, 0) for result in cv_results['fold_results']]
            ax.plot(fold_numbers, val_values, 'o-', label='验证集', linewidth=2, markersize=8)
            
            # 测试集指标（如果有）
            if cv_results['test_metrics_mean']:
                test_values = [result['test_metrics'].get(metric, 0) for result in cv_results['fold_results'] 
                              if result['test_metrics']]
                if test_values:
                    ax.plot(fold_numbers[:len(test_values)], test_values, 's-', label='测试集', linewidth=2, markersize=8)
            
            # 添加均值线
            val_mean = cv_results['val_metrics_mean'].get(metric, 0)
            ax.axhline(y=val_mean, color='blue', linestyle='--', alpha=0.7, label=f'验证均值: {val_mean:.4f}')
            
            if cv_results['test_metrics_mean']:
                test_mean = cv_results['test_metrics_mean'].get(metric, 0)
                ax.axhline(y=test_mean, color='orange', linestyle='--', alpha=0.7, label=f'测试均值: {test_mean:.4f}')
            
            ax.set_xlabel('折数')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} - {cv_type} {n_splits}折交叉验证')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(fold_numbers)
        
        plt.tight_layout()
        plot_file = results_dir / f"cv_metrics_{cv_type}_{n_splits}fold.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"交叉验证图表已保存: {plot_file}")
        
        # 绘制指标分布箱线图
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        metrics_data = []
        metrics_labels = []
        
        for metric in main_metrics:
            val_values = [result['val_metrics'].get(metric, 0) for result in cv_results['fold_results']]
            metrics_data.append(val_values)
            metrics_labels.append(f'{metric.upper()}\n(验证)')
            
            if cv_results['test_metrics_mean']:
                test_values = [result['test_metrics'].get(metric, 0) for result in cv_results['fold_results'] 
                              if result['test_metrics']]
                if test_values:
                    metrics_data.append(test_values)
                    metrics_labels.append(f'{metric.upper()}\n(测试)')
        
        ax.boxplot(metrics_data, labels=metrics_labels)
        ax.set_title(f'交叉验证指标分布 - {cv_type} {n_splits}折')
        ax.set_ylabel('指标值')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        boxplot_file = results_dir / f"cv_distribution_{cv_type}_{n_splits}fold.png"
        plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"交叉验证分布图已保存: {boxplot_file}")
