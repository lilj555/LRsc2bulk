"""
模型评估器
"""
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, metrics: List[str]):
        self.metrics = metrics
        self.thresholds = None
        self.supported_metrics = {
            'mse': self._calculate_mse,
            'mae': self._calculate_mae,
            'rmse': self._calculate_rmse,
            'r2': self._calculate_r2,
            'pearson': self._calculate_pearson,
            'spearman': self._calculate_spearman,
            'explained_variance': self._calculate_explained_variance,
            # classification metrics
            'accuracy_micro': self._accuracy_micro,
            'accuracy_macro': self._accuracy_macro,
            'f1_micro': self._f1_micro,
            'f1_macro': self._f1_macro,
            'roc_auc_micro': self._roc_auc_micro,
            'roc_auc_macro': self._roc_auc_macro,
            'precision_micro': self._precision_micro,
            'precision_macro': self._precision_macro,
            'recall_micro': self._recall_micro,
            'recall_macro': self._recall_macro
        }
        
        # 验证指标
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"不支持的评估指标: {metric}")
    
    def _calculate_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算均方误差"""
        return mean_squared_error(y_true, y_pred)
    
    def _calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算平均绝对误差"""
        return mean_absolute_error(y_true, y_pred)
    
    def _calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算均方根误差"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算R²分数"""
        return r2_score(y_true, y_pred)
    
    def _calculate_pearson(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算Pearson相关系数"""
        # 展平数组以计算整体相关性
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # 移除NaN值
        mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
        if np.sum(mask) < 2:
            return 0.0
        
        corr, _ = pearsonr(y_true_flat[mask], y_pred_flat[mask])
        return corr if not np.isnan(corr) else 0.0
    
    def _calculate_spearman(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算Spearman相关系数"""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
        if np.sum(mask) < 2:
            return 0.0
        
        corr, _ = spearmanr(y_true_flat[mask], y_pred_flat[mask])
        return corr if not np.isnan(corr) else 0.0
    
    def _calculate_explained_variance(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算解释方差分数"""
        from sklearn.metrics import explained_variance_score
        return explained_variance_score(y_true, y_pred)
    
    def evaluate(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
        """
        评估模型性能
        Args:
            y_pred: 预测值 [batch_size, num_targets]
            y_true: 真实值 [batch_size, num_targets]
        Returns:
            评估指标字典
        """
        # 转换为numpy数组
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        
        results = {}
        threshold_metrics = {
            'accuracy_micro','accuracy_macro','f1_micro','f1_macro',
            'precision_micro','precision_macro','recall_micro','recall_macro'
        }
        
        for metric in self.metrics:
            try:
                if metric in threshold_metrics:
                    value = self.supported_metrics[metric](y_true, y_pred, threshold=self.thresholds if self.thresholds is not None else 0.5)
                else:
                    value = self.supported_metrics[metric](y_true, y_pred)
                results[metric] = float(value)
            except Exception as e:
                logger.warning(f"计算指标{metric}时出错: {e}")
                results[metric] = 0.0
        
        return results
    
    def evaluate_per_target(self, y_pred: torch.Tensor, y_true: torch.Tensor, 
                           target_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        按目标变量评估性能
        Args:
            y_pred: 预测值 [batch_size, num_targets]
            y_true: 真实值 [batch_size, num_targets]
            target_names: 目标变量名称列表
        Returns:
            每个目标的评估结果DataFrame
        """
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        
        num_targets = y_pred.shape[1]
        if target_names is None:
            target_names = [f"target_{i}" for i in range(num_targets)]
        
        results = []
        
        for i in range(num_targets):
            target_result = {'target': target_names[i]}
            
            for metric in self.metrics:
                try:
                    value = self.supported_metrics[metric](y_true[:, i], y_pred[:, i])
                    target_result[metric] = float(value)
                except Exception as e:
                    logger.warning(f"计算目标{target_names[i]}的指标{metric}时出错: {e}")
                    target_result[metric] = 0.0
            
            results.append(target_result)
        
        return pd.DataFrame(results)
    
    def plot_predictions(self, y_pred: torch.Tensor, y_true: torch.Tensor, 
                        save_path: Optional[Path] = None, 
                        target_names: Optional[List[str]] = None,
                        max_targets: int = 9) -> None:
        """
        绘制预测结果图（英文）
        Args:
            y_pred: 预测值
            y_true: 真实值
            save_path: 保存路径
            target_names: 目标变量名称
            max_targets: 最大显示目标数量
        """
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        
        num_targets = min(y_pred.shape[1], max_targets)
        
        # 计算子图布局
        n_cols = min(3, num_targets)
        n_rows = (num_targets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if num_targets == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # 检测是否为二分类（0/1标签）
        unique_vals = np.unique(y_true)
        is_binary = np.array_equal(unique_vals, np.array([0, 1])) or np.array_equal(unique_vals, np.array([0])) or np.array_equal(unique_vals, np.array([1]))

        for i in range(num_targets):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            if is_binary:
                # 二分类：画真实标签 vs 概率散点
                ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=2)
                ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.8)
                target_name = target_names[i] if target_names else f"Target {i}"
                ax.set_title(f"{target_name} (Binary)\nProb by True label")
                ax.set_xlabel("True Label")
                ax.set_ylabel("Predicted Probability")
            else:
                # 回归：散点 + 对角线
                ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=2)
                min_val = min(y_true[:, i].min(), y_pred[:, i].min())
                max_val = max(y_true[:, i].max(), y_pred[:, i].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                corr = self._calculate_pearson(y_true[:, i], y_pred[:, i])
                r2 = self._calculate_r2(y_true[:, i], y_pred[:, i])
                target_name = target_names[i] if target_names else f"Target {i}"
                ax.set_title(f"{target_name}\nR²={r2:.3f}, r={corr:.3f}")
                ax.set_xlabel("True Value")
                ax.set_ylabel("Predicted Value")
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(num_targets, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction plot saved to: {save_path}")
        
        plt.close()
    
    def plot_residuals(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                      save_path: Optional[Path] = None) -> None:
        """
        绘制残差图（英文）
        Args:
            y_pred: 预测值
            y_true: 真实值
            save_path: 保存路径
        """
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 残差 vs 预测值
        axes[0, 0].scatter(y_pred.flatten(), residuals.flatten(), alpha=0.6, s=1)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel("Predicted Value")
        axes[0, 0].set_ylabel("Residual")
        axes[0, 0].set_title("Residuals vs Predicted")
        axes[0, 0].grid(True, alpha=0.3)
        
        # 残差直方图
        axes[0, 1].hist(residuals.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel("Residual")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Residual Distribution")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q图
        stats.probplot(residuals.flatten(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Residual Q-Q Plot")
        axes[1, 0].grid(True, alpha=0.3)
        
        # 残差的绝对值 vs 预测值
        axes[1, 1].scatter(y_pred.flatten(), np.abs(residuals.flatten()), alpha=0.6, s=1)
        axes[1, 1].set_xlabel("Predicted Value")
        axes[1, 1].set_ylabel("|Residual|")
        axes[1, 1].set_title("Absolute Residual vs Predicted")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residual plot saved to: {save_path}")
        
        plt.close()
    
    def generate_report(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                       target_names: Optional[List[str]] = None,
                       save_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        生成完整的评估报告
        Args:
            y_pred: 预测值
            y_true: 真实值
            target_names: 目标变量名称
            save_dir: 保存目录
        Returns:
            评估报告字典
        """
        logger.info("生成评估报告...")
        
        # 整体评估
        overall_metrics = self.evaluate(y_pred, y_true)
        
        # 按目标评估
        per_target_metrics = self.evaluate_per_target(y_pred, y_true, target_names)
        
        # 统计信息
        if isinstance(y_pred, torch.Tensor):
            y_pred_np = y_pred.detach().cpu().numpy()
        else:
            y_pred_np = y_pred
        if isinstance(y_true, torch.Tensor):
            y_true_np = y_true.detach().cpu().numpy()
        else:
            y_true_np = y_true
        
        statistics = {
            'num_samples': y_pred_np.shape[0],
            'num_targets': y_pred_np.shape[1],
            'pred_mean': float(np.mean(y_pred_np)),
            'pred_std': float(np.std(y_pred_np)),
            'true_mean': float(np.mean(y_true_np)),
            'true_std': float(np.std(y_true_np)),
            'pred_min': float(np.min(y_pred_np)),
            'pred_max': float(np.max(y_pred_np)),
            'true_min': float(np.min(y_true_np)),
            'true_max': float(np.max(y_true_np))
        }
        
        report = {
            'overall_metrics': overall_metrics,
            'per_target_metrics': per_target_metrics,
            'statistics': statistics
        }
        
        # 保存报告
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存数值结果
            report_path = save_dir / 'evaluation_report.json'
            import json
            with open(report_path, 'w', encoding='utf-8') as f:
                # 转换DataFrame为字典以便JSON序列化
                report_copy = report.copy()
                report_copy['per_target_metrics'] = per_target_metrics.to_dict('records')
                json.dump(report_copy, f, indent=2, ensure_ascii=False)
            
            # 保存CSV
            per_target_metrics.to_csv(save_dir / 'per_target_metrics.csv', index=False)
            
            # 生成图表（英文）
            self.plot_predictions(y_pred, y_true, save_dir / 'predictions.png', target_names)
            # 如果是分类任务（存在分类指标），跳过残差图
            classification_metrics = { 'accuracy_micro','accuracy_macro','f1_micro','f1_macro','roc_auc_micro','roc_auc_macro' }
            if not any(m in classification_metrics for m in self.metrics):
                self.plot_residuals(y_pred, y_true, save_dir / 'residuals.png')
            
            logger.info(f"Evaluation report saved to: {save_dir}")
        
        return report

    # ---------------------- Classification metrics ----------------------
    def _threshold_predictions(self, y_pred: np.ndarray, threshold: float | np.ndarray = 0.5) -> np.ndarray:
        if isinstance(threshold, np.ndarray):
            if y_pred.ndim == 1:
                t = threshold[0] if threshold.ndim == 1 else float(0.5)
                return (y_pred >= t).astype(int)
            else:
                if threshold.ndim == 1 and threshold.shape[0] == y_pred.shape[1]:
                    return (y_pred >= threshold.reshape(1, -1)).astype(int)
        return (y_pred >= threshold).astype(int)

    def _accuracy_micro(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        # 跳过全零样本（该样本所有目标均为0）
        if y_true.ndim == 1:
            mask = y_true > 0
        else:
            mask = y_true.sum(axis=1) > 0
        if not np.any(mask):
            return 0.0
        y_true_m = y_true[mask]
        y_pred_m = y_pred[mask]
        y_pred_bin = self._threshold_predictions(y_pred_m, threshold)
        return float((y_pred_bin == y_true_m).mean())

    def _accuracy_macro(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        # 跳过全零样本（行级）
        if y_true.ndim == 1:
            mask = y_true > 0
            if not np.any(mask):
                return 0.0
            y_true_m = y_true[mask]
            y_pred_m = y_pred[mask]
            y_pred_bin = self._threshold_predictions(y_pred_m, threshold)
            return float((y_pred_bin == y_true_m).mean())
        else:
            mask = y_true.sum(axis=1) > 0
            if not np.any(mask):
                return 0.0
            y_true_m = y_true[mask]
            y_pred_m = y_pred[mask]
            y_pred_bin = self._threshold_predictions(y_pred_m, threshold)
            per_target_acc = (y_pred_bin == y_true_m).mean(axis=0)
            return float(np.mean(per_target_acc))

    def _f1_micro(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        from sklearn.metrics import f1_score
        # 跳过全零样本（行级）
        if y_true.ndim == 1:
            mask = y_true > 0
        else:
            mask = y_true.sum(axis=1) > 0
        if not np.any(mask):
            return 0.0
        y_true_m = y_true[mask]
        y_pred_m = y_pred[mask]
        y_pred_bin = self._threshold_predictions(y_pred_m, threshold)
        try:
            return float(f1_score(y_true_m.flatten(), y_pred_bin.flatten()))
        except Exception:
            return 0.0

    def _f1_macro(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        from sklearn.metrics import f1_score
        # 跳过全零样本（行级），并跳过单一类别目标
        if y_true.ndim == 1:
            mask = y_true > 0
            if not np.any(mask):
                return 0.0
            yt = y_true[mask]
            yp = self._threshold_predictions(y_pred[mask], threshold)
            try:
                return float(f1_score(yt, yp))
            except Exception:
                return 0.0
        else:
            mask = y_true.sum(axis=1) > 0
            if not np.any(mask):
                return 0.0
            y_true_m = y_true[mask]
            y_pred_bin = self._threshold_predictions(y_pred[mask], threshold)
            f1s = []
            for i in range(y_true_m.shape[1]):
                yt = y_true_m[:, i]
                yp = y_pred_bin[:, i]
                if len(np.unique(yt)) < 2:
                    continue
                try:
                    f1s.append(f1_score(yt, yp))
                except Exception:
                    continue
            return float(np.mean(f1s)) if f1s else 0.0

    def _roc_auc_micro(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score
        # 跳过全零样本（行级）
        if y_true.ndim == 1:
            mask = y_true > 0
        else:
            mask = y_true.sum(axis=1) > 0
        if not np.any(mask):
            return 0.0
        y_true_m = y_true[mask]
        y_pred_m = y_pred[mask]
        try:
            return float(roc_auc_score(y_true_m.flatten(), y_pred_m.flatten()))
        except Exception:
            return 0.0

    def _roc_auc_macro(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score
        # 支持 1D（单目标）与 2D（多目标）输入；并跳过全零样本（行级）
        if y_true.ndim == 1:
            mask = y_true > 0
            if not np.any(mask):
                return 0.0
            yt = y_true[mask]
            yp = y_pred[mask]
            if len(np.unique(yt)) < 2:
                return 0.0
            try:
                auc = roc_auc_score(yt, yp)
                return float(auc) if not np.isnan(auc) else 0.0
            except Exception:
                return 0.0
        else:
            mask = y_true.sum(axis=1) > 0
            if not np.any(mask):
                return 0.0
            y_true_m = y_true[mask]
            y_pred_m = y_pred[mask]
            aucs = []
            for i in range(y_true_m.shape[1]):
                yt = y_true_m[:, i]
                yp = y_pred_m[:, i]
                if len(np.unique(yt)) < 2:
                    continue
                try:
                    auc = roc_auc_score(yt, yp)
                    if not np.isnan(auc):
                        aucs.append(auc)
                except Exception:
                    continue
            return float(np.mean(aucs)) if aucs else 0.0

    def _precision_micro(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        from sklearn.metrics import precision_score
        # 跳过全零样本（行级）
        if y_true.ndim == 1:
            mask = y_true > 0
        else:
            mask = y_true.sum(axis=1) > 0
        if not np.any(mask):
            return 0.0
        y_true_m = y_true[mask]
        y_pred_m = y_pred[mask]
        y_pred_bin = self._threshold_predictions(y_pred_m, threshold)
        try:
            return float(precision_score(y_true_m.flatten(), y_pred_bin.flatten()))
        except Exception:
            return 0.0

    def _precision_macro(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        from sklearn.metrics import precision_score
        # 跳过全零样本（行级）
        if y_true.ndim == 1:
            mask = y_true > 0
            if not np.any(mask):
                return 0.0
            yt = y_true[mask]
            yp = self._threshold_predictions(y_pred[mask], threshold)
            try:
                return float(precision_score(yt, yp))
            except Exception:
                return 0.0
        else:
            mask = y_true.sum(axis=1) > 0
            if not np.any(mask):
                return 0.0
            y_true_m = y_true[mask]
            y_pred_m = self._threshold_predictions(y_pred[mask], threshold)
            precisions = []
            for i in range(y_true_m.shape[1]):
                yt = y_true_m[:, i]
                yp = y_pred_m[:, i]
                if len(np.unique(yt)) < 2:
                    continue
                try:
                    precisions.append(precision_score(yt, yp))
                except Exception:
                    continue
            return float(np.mean(precisions)) if precisions else 0.0

    def _recall_micro(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        from sklearn.metrics import recall_score
        # 跳过全零样本（行级）
        if y_true.ndim == 1:
            mask = y_true > 0
        else:
            mask = y_true.sum(axis=1) > 0
        if not np.any(mask):
            return 0.0
        y_true_m = y_true[mask]
        y_pred_m = y_pred[mask]
        y_pred_bin = self._threshold_predictions(y_pred_m, threshold)
        try:
            return float(recall_score(y_true_m.flatten(), y_pred_bin.flatten()))
        except Exception:
            return 0.0

    def _recall_macro(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        from sklearn.metrics import recall_score
        # 跳过全零样本（行级）
        if y_true.ndim == 1:
            mask = y_true > 0
            if not np.any(mask):
                return 0.0
            yt = y_true[mask]
            yp = self._threshold_predictions(y_pred[mask], threshold)
            try:
                return float(recall_score(yt, yp))
            except Exception:
                return 0.0
        else:
            mask = y_true.sum(axis=1) > 0
            if not np.any(mask):
                return 0.0
            y_true_m = y_true[mask]
            y_pred_m = self._threshold_predictions(y_pred[mask], threshold)
            recalls = []
            for i in range(y_true_m.shape[1]):
                yt = y_true_m[:, i]
                yp = y_pred_m[:, i]
                if len(np.unique(yt)) < 2:
                    continue
                try:
                    recalls.append(recall_score(yt, yp))
                except Exception:
                    continue
            return float(np.mean(recalls)) if recalls else 0.0
