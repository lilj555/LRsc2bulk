"""
GNN模型训练器
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wandb
import sys

from ..utils.gpu_utils import GPUManager, setup_mixed_precision, get_optimal_batch_size
from ..evaluation.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        检查是否应该早停
        Returns:
            True if should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
                logger.info("恢复最佳权重")
            return True
        return False

class LearningRateScheduler:
    """学习率调度器"""
    
    def __init__(self, optimizer: optim.Optimizer, scheduler_type: str = "cosine",
                 warmup_epochs: int = 10, total_epochs: int = 100, 
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.initial_lr = optimizer.param_groups[0]['lr']
        
        if scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr
            )
        elif scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=total_epochs // 3, gamma=0.1
            )
        elif scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, min_lr=min_lr
            )
        else:
            self.scheduler = None
    
    def step(self, epoch: int, val_loss: Optional[float] = None):
        """更新学习率"""
        if epoch < self.warmup_epochs:
            # Warmup阶段
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # 正常调度
            if self.scheduler_type == "plateau" and val_loss is not None:
                self.scheduler.step(val_loss)
            elif self.scheduler:
                self.scheduler.step()

class GNNTrainer:
    """GNN模型训练器"""
    
    def __init__(self, model: nn.Module, config: dict, metadata: dict):
        self.model = model
        self.config = config
        self.metadata = metadata
        self.task_type = config.get('task_type', 'regression')
        self.pred_threshold = float(config.get('classification', {}).get('pred_threshold', 0.5))
        
        # GPU管理
        self.gpu_manager = GPUManager()
        self.device = self.gpu_manager.setup_device(
            device_ids=config['gpu']['device_ids'],
            min_memory_gb=2.0
        )
        self.model.to(self.device)
        self.is_data_parallel = False
        self.device_ids = None
        if self.device.type == 'cuda':
            try:
                visible_count = torch.cuda.device_count()
                if visible_count > 1:
                    cfg_ids = config.get('gpu', {}).get('device_ids', list(range(visible_count)))
                    if not isinstance(cfg_ids, list) or len(cfg_ids) == 0:
                        cfg_ids = list(range(visible_count))
                    avail = getattr(self.gpu_manager, 'available_gpus', [])
                    if isinstance(avail, list) and len(avail) > 0:
                        filt = [i for i in cfg_ids if i in avail]
                    else:
                        filt = cfg_ids
                    if self.device.index is not None and self.device.index not in filt:
                        filt = [self.device.index] + [i for i in filt if i != self.device.index]
                    if len(filt) > 1:
                        self.model = nn.DataParallel(self.model, device_ids=filt)
                        self.is_data_parallel = True
                        self.device_ids = filt
                        logger.info(f"已启用DataParallel，使用GPU: {filt}")
            except Exception as e:
                logger.warning(f"启用DataParallel失败，回退到单GPU: {e}")
        if not self.is_data_parallel and self.device.type == 'cuda':
            self.device_ids = [self.device.index]
        
        # 混合精度训练
        self.use_amp, self.scaler = setup_mixed_precision()
        if config['gpu']['mixed_precision'] and not self.use_amp:
            logger.warning("混合精度训练不可用，使用FP32训练")
        self.use_amp = self.use_amp and config['gpu']['mixed_precision']
        
        # 优化器
        self.optimizer = self._create_optimizer()

        # 损失函数
        if self.task_type == 'classification':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()
        
        # 学习率调度器
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer,
            scheduler_type=config['scheduler']['type'],
            warmup_epochs=config['scheduler']['warmup_epochs'],
            total_epochs=config['training']['epochs'],
            min_lr=float(config['scheduler']['min_lr'])  # 确保是浮点数
        )
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=config['training']['patience'],
            min_delta=config['training']['min_delta']
        )
        
        # 评估器
        self.evaluator = ModelEvaluator(config['evaluation']['metrics'])
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'gpu_memory': []
        }
        
        # 输出目录
        self.model_dir = Path(config['output']['model_dir'])
        self.log_dir = Path(config['output']['log_dir'])
        self.result_dir = Path(config['output']['result_dir'])
        
        for dir_path in [self.model_dir, self.log_dir, self.result_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        optimizer_config = self.config['optimizer']
        training_config = self.config['training']
        
        if optimizer_config['type'].lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                betas=optimizer_config['betas'],
                eps=optimizer_config['eps'],
                weight_decay=training_config['weight_decay']
            )
        elif optimizer_config['type'].lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                betas=optimizer_config['betas'],
                eps=optimizer_config['eps'],
                weight_decay=training_config['weight_decay']
            )
        elif optimizer_config['type'].lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                momentum=0.9,
                weight_decay=training_config['weight_decay']
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_config['type']}")
    
    def _model_state_dict(self):
        if self.is_data_parallel:
            return self.model.module.state_dict()
        return self.model.state_dict()
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """将批次数据移动到设备"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if self.is_data_parallel:
                    # DataParallel按batch维度scatter输入；为避免设备不匹配，仅将目标张量移动到主设备用于聚合后的损失计算
                    if key == 'lr_scores':
                        device_batch[key] = value.to(self.device, non_blocking=True)
                    else:
                        device_batch[key] = value
                else:
                    device_batch[key] = value.to(self.device, non_blocking=True)
            else:
                device_batch[key] = value
        return device_batch
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(
            train_loader,
            desc="Train",
            leave=False,
            disable=not sys.stdout.isatty()
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            batch = self._move_batch_to_device(batch)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast('cuda'):
                    predictions = self.model(batch)
                    loss = self.criterion(predictions, batch['lr_scores'])
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.config['training']['gradient_clip'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(batch)
                loss = self.criterion(predictions, batch['lr_scores'])
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                if self.config['training']['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
            
            # 清理GPU缓存
            if batch_idx % 100 == 0:
                self.gpu_manager.clear_cache()
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float], torch.Tensor, torch.Tensor]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc="Valid",
                leave=False,
                disable=not sys.stdout.isatty()
            ):
                batch = self._move_batch_to_device(batch)
                
                if self.use_amp:
                    with autocast('cuda'):
                        predictions = self.model(batch)
                        loss = self.criterion(predictions, batch['lr_scores'])
                else:
                    predictions = self.model(batch)
                    loss = self.criterion(predictions, batch['lr_scores'])
                
                total_loss += loss.item()
                # 分类任务下，保存为概率以便评估AUC/F1等
                if self.task_type == 'classification':
                    probs = torch.sigmoid(predictions).cpu()
                    all_predictions.append(probs)
                else:
                    all_predictions.append(predictions.cpu())
                all_targets.append(batch['lr_scores'].cpu())
        
        # 计算评估指标
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.evaluator.evaluate(all_predictions, all_targets)
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss, metrics, all_predictions, all_targets
    
    def save_checkpoint(self, epoch: int, val_loss: float, metrics: Dict[str, float]):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self._model_state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'config': self.config,
            'metadata': self.metadata,
            'history': self.history
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 保存最新检查点
        checkpoint_path = self.model_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if not hasattr(self, 'best_val_loss') or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_path = self.model_dir / 'best_model.pth'
            torch.save(checkpoint, best_model_path)
            logger.info(f"保存最佳模型，验证损失: {val_loss:.4f}")
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 学习率曲线
        axes[0, 1].plot(self.history['learning_rate'])
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True)
        
        # GPU内存使用
        if self.history['gpu_memory']:
            axes[1, 0].plot(self.history['gpu_memory'])
            axes[1, 0].set_title('GPU Memory Usage')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Memory (GB)')
            axes[1, 0].grid(True)
        
        # 验证指标（如果有的话）
        if hasattr(self, 'val_metrics_history'):
            for metric_name, values in self.val_metrics_history.items():
                axes[1, 1].plot(values, label=metric_name)
            axes[1, 1].set_title('Validation Metrics')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Metric Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.result_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, start_epoch: int = 0) -> Dict[str, Any]:
        """完整训练流程"""
        logger.info("开始训练...")
        logger.info(f"混合精度: {self.use_amp}")
        logger.info(f"模型参数数: {sum(p.numel() for p in self.model.parameters()):,}")
        
        if start_epoch > 0:
            logger.info(f"从第 {start_epoch + 1} 轮开始恢复训练")
        
        # 初始化wandb（如果配置了的话）
        if hasattr(self.config, 'wandb') and self.config.get('wandb', {}).get('enabled', False):
            wandb.init(
                project=self.config['wandb']['project'],
                config=self.config,
                name=self.config['wandb'].get('run_name', 'gnn_lr_prediction')
            )
            wandb.watch(self.model)
        
        start_time = time.time()
        self.val_metrics_history = {}
        # 分类任务：根据训练集不平衡动态设置正类权重
        if self.task_type == 'classification' and bool(self.config.get('classification', {}).get('use_pos_weight_loss', False)):
            try:
                pos_weight = self._compute_pos_weight(train_loader)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
                logger.info("已启用按目标的正类加权损失")
            except Exception as e:
                logger.warning(f"计算正类权重失败，继续使用默认损失: {e}")
        
        for epoch in range(start_epoch, self.config['training']['epochs']):
            epoch_start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_metrics, val_pred, val_true = self.validate_epoch(val_loader)
            
            # 动态阈值计算（如果启用）
            use_dynamic_threshold = self.config.get('classification', {}).get('use_dynamic_threshold', True)
            if self.task_type == 'classification':
                if use_dynamic_threshold:
                    try:
                        self._compute_and_save_per_target_thresholds(
                            epoch, val_pred, val_true, self.metadata.get('lr_pairs')
                        )
                        logger.info("已计算并缓存逐LR阈值")
                    except Exception as e:
                        logger.warning(f"逐LR阈值计算失败: {e}")
                else:
                    # 使用统一阈值
                    self.per_target_thresholds = [self.pred_threshold] * val_pred.shape[1]
                    # logger.info(f"使用统一阈值: {self.pred_threshold}")

            # 将逐目标阈值注入评估器，使micro/macro使用调优阈值
            if hasattr(self.evaluator, 'thresholds') and hasattr(self, 'per_target_thresholds'):
                import numpy as np
                self.evaluator.thresholds = np.array(self.per_target_thresholds)
                try:
                    val_metrics = self.evaluator.evaluate(val_pred, val_true)
                except Exception:
                    pass
            
            # 更新学习率
            self.lr_scheduler.step(epoch, val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # 记录GPU内存使用
            memory_info = self.gpu_manager.get_memory_usage()
            if memory_info:
                self.history['gpu_memory'].append(memory_info['allocated'])
                util_info = self.gpu_manager.get_gpu_utilization()
                if util_info is not None:
                    try:
                        if self.device_ids:
                            for did in self.device_ids:
                                ui = util_info.get(did, None)
                                if ui:
                                    logger.info(
                                        f"GPU{did} 利用率 {ui['utilization_percent']:.0f}% | 显存 {ui['memory_used_gb']:.1f}/{ui['memory_total_gb']:.1f}GB"
                                    )
                    except Exception:
                        pass
            
            # 记录验证指标
            for metric_name, metric_value in val_metrics.items():
                if metric_name not in self.val_metrics_history:
                    self.val_metrics_history[metric_name] = []
                self.val_metrics_history[metric_name].append(metric_value)
            
            # 计算epoch时间
            epoch_time = time.time() - epoch_start_time
            
            # 日志输出
            logger.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, "
                f"学习率: {current_lr:.2e}, 时间: {epoch_time:.1f}s"
            )
            
            # 输出验证/测试指标保持一致的格式
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            logger.info(f"验证指标 - {metrics_str}")

            try:
                if bool(self.config.get('evaluation', {}).get('generate_lr_scatter', False)):
                    self._save_per_lr_plots(epoch, val_pred, val_true, self.metadata.get('lr_pairs'))
            except Exception as e:
                logger.warning(f"保存每LR可视化时出错: {e}")

            # 每epoch可视化
            try:
                self._plot_epoch_metrics_dashboard(epoch, val_metrics)
                if self.task_type == 'classification':
                    self._plot_classification_curves(epoch, val_pred, val_true)
                    self._plot_confusion_and_threshold_sweep(epoch, val_pred, val_true)
                    self._save_per_lr_classification_visuals(epoch, val_pred, val_true, self.metadata.get('lr_pairs'))
            except Exception as e:
                logger.warning(f"生成epoch可视化时出错: {e}")
            
            # wandb记录
            if hasattr(self.config, 'wandb') and self.config.get('wandb', {}).get('enabled', False):
                log_dict = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': current_lr,
                    'epoch_time': epoch_time
                }
                log_dict.update({f'val_{k}': v for k, v in val_metrics.items()})
                if memory_info:
                    log_dict['gpu_memory_gb'] = memory_info['allocated']
                wandb.log(log_dict)
            
            self.save_checkpoint(epoch, val_loss, val_metrics)
            
            # 早停检查
            if self.early_stopping(val_loss, self.model):
                logger.info(f"早停触发，在第{epoch+1}个epoch停止训练")
                break
        
        # 训练完成
        total_time = time.time() - start_time
        logger.info(f"训练完成，总时间: {total_time/3600:.2f}小时")
        
        # 保存最终模型
        self.save_checkpoint(epoch, val_loss, val_metrics)
        
        # 绘制训练历史
        self.plot_training_history()
        
        # 返回训练结果
        return {
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'final_metrics': val_metrics,
            'total_epochs': epoch + 1,
            'total_time': total_time,
            'best_val_loss': getattr(self, 'best_val_loss', val_loss)
        }

    def _ensure_epoch_dir(self, epoch: int) -> Path:
        d = self.result_dir / f'epoch_{epoch+1}'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _plot_epoch_metrics_dashboard(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        epoch_dir = self._ensure_epoch_dir(epoch)
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(self.history['learning_rate'])
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('LR')
        axes[0, 1].grid(True, alpha=0.3)

        groups = [
            ('accuracy_micro', 'accuracy_macro'),
            ('f1_micro', 'f1_macro'),
            ('precision_micro', 'precision_macro'),
            ('recall_micro', 'recall_macro'),
            ('roc_auc_micro', 'roc_auc_macro')
        ]
        names = []
        micro_vals = []
        macro_vals = []
        for m_key, M_key in groups:
            if m_key in val_metrics and M_key in val_metrics:
                names.append(m_key.split('_')[0])
                micro_vals.append(val_metrics[m_key])
                macro_vals.append(val_metrics[M_key])
        x = np.arange(len(names))
        width = 0.35
        axes[1, 0].bar(x - width/2, micro_vals, width, label='micro')
        axes[1, 0].bar(x + width/2, macro_vals, width, label='macro')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(names)
        axes[1, 0].set_ylim(0.0, 1.0)
        axes[1, 0].set_title('Classification Metrics')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        if self.history['gpu_memory']:
            axes[1, 1].plot(self.history['gpu_memory'])
            axes[1, 1].set_title('GPU Memory (GB)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('GB')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(epoch_dir / 'metrics_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_classification_curves(self, epoch: int, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        epoch_dir = self._ensure_epoch_dir(epoch)
        yp = y_pred.detach().cpu().numpy()
        yt = y_true.detach().cpu().numpy()
        if yt.ndim == 1:
            mask = yt > 0
        else:
            mask = yt.sum(axis=1) > 0
        if not np.any(mask):
            return
        yt_m = yt[mask]
        yp_m = yp[mask]
        yt_flat = yt_m.flatten()
        yp_flat = yp_m.flatten()

        try:
            from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
            fpr, tpr, _ = roc_curve(yt_flat, yp_flat)
            roc_auc = auc(fpr, tpr)
            prec, rec, _ = precision_recall_curve(yt_flat, yp_flat)
            ap = average_precision_score(yt_flat, yp_flat)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
            axes[0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
            axes[0].set_title('Micro ROC')
            axes[0].set_xlabel('FPR')
            axes[0].set_ylabel('TPR')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(rec, prec, label=f'AP={ap:.3f}')
            axes[1].set_title('Micro PR')
            axes[1].set_xlabel('Recall')
            axes[1].set_ylabel('Precision')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(epoch_dir / 'curves_micro.png', dpi=300, bbox_inches='tight')
            plt.close()

            fig = plt.figure(figsize=(6,4))
            plt.hist(yp_flat, bins=50, alpha=0.8, edgecolor='black')
            plt.title('Predicted Probability Distribution')
            plt.xlabel('Probability')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(epoch_dir / 'prob_hist.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception:
            return

    def _plot_confusion_and_threshold_sweep(self, epoch: int, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        epoch_dir = self._ensure_epoch_dir(epoch)
        yp = y_pred.detach().cpu().numpy()
        yt = y_true.detach().cpu().numpy()
        if yt.ndim == 1:
            mask = yt > 0
        else:
            mask = yt.sum(axis=1) > 0
        if not np.any(mask):
            return
        yt_m = yt[mask]
        yp_m = yp[mask]
        yt_flat = yt_m.flatten()
        yp_flat = yp_m.flatten()
        try:
            from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
            # Confusion at configured threshold
            yp_bin = (yp_flat >= self.pred_threshold).astype(int)
            cm = confusion_matrix(yt_flat, yp_bin)
            # Threshold sweep curves
            thresholds = np.linspace(0.05, 0.95, 19)
            precs, recs, f1s = [], [], []
            for th in thresholds:
                yb = (yp_flat >= th).astype(int)
                try:
                    precs.append(precision_score(yt_flat, yb))
                    recs.append(recall_score(yt_flat, yb))
                    f1s.append(f1_score(yt_flat, yb))
                except Exception:
                    precs.append(0.0)
                    recs.append(0.0)
                    f1s.append(0.0)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            # Confusion matrix heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
            axes[0].set_title(f'Confusion @ th={self.pred_threshold}')
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('True')
            # Threshold sweep
            axes[1].plot(thresholds, precs, label='Precision')
            axes[1].plot(thresholds, recs, label='Recall')
            axes[1].plot(thresholds, f1s, label='F1')
            axes[1].axvline(self.pred_threshold, color='k', linestyle='--', alpha=0.6, label='current th')
            axes[1].set_title('Threshold Sweep (micro)')
            axes[1].set_xlabel('Threshold')
            axes[1].set_ylabel('Score')
            axes[1].set_ylim(0.0, 1.0)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(epoch_dir / 'confusion_threshold.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception:
            return

    def _compute_pos_weight(self, train_loader: DataLoader) -> torch.Tensor:
        # 汇总训练集每个目标的正负样本数量
        num_targets = train_loader.dataset.lr_scores.shape[1]
        pos_counts = torch.zeros(num_targets, dtype=torch.float32)
        total_counts = 0
        for i in range(len(train_loader.dataset)):
            labels = train_loader.dataset.lr_scores[i]
            pos_counts += labels.float()
            total_counts += 1
        neg_counts = total_counts - pos_counts
        # pos_weight = neg/pos，避免除零并限制范围
        pos_counts = torch.clamp(pos_counts, min=1.0)
        pos_weight = neg_counts / pos_counts
        pos_weight = torch.clamp(pos_weight, min=1.0, max=100.0)
        return pos_weight

    def _compute_and_save_per_target_thresholds(self, epoch: int, y_pred: torch.Tensor, y_true: torch.Tensor, target_names: Optional[List[str]]) -> None:
        epoch_dir = self._ensure_epoch_dir(epoch)
        import numpy as np
        from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve
        yp = y_pred.detach().cpu().numpy()
        yt = y_true.detach().cpu().numpy()
        num_targets = yp.shape[1]
        thresholds = np.full(num_targets, self.pred_threshold, dtype=np.float32)
        tr_cfg = self.config.get('classification', {}).get('threshold_range', None)
        obj = str(self.config.get('classification', {}).get('threshold_objective', 'f1')).lower()
        beta = float(self.config.get('classification', {}).get('fbeta', 1.0))
        if isinstance(tr_cfg, (list, tuple)) and len(tr_cfg) == 3:
            t_min, t_max, t_num = float(tr_cfg[0]), float(tr_cfg[1]), int(tr_cfg[2])
            sweep = np.linspace(t_min, t_max, t_num)
        else:
            sweep = np.linspace(0.05, 0.95, 19)
        for i in range(num_targets):
            yti = yt[:, i]
            ypi = yp[:, i]
            if np.unique(yti).size < 2:
                continue
            best_score = -1.0
            best_th = thresholds[i]
            for th in sweep:
                yb = (ypi >= th).astype(int)
                try:
                    if obj == 'fbeta':
                        p = precision_score(yti, yb, zero_division=0)
                        r = recall_score(yti, yb, zero_division=0)
                        denom = (beta * beta * p + r)
                        score_obj = ((1 + beta * beta) * p * r / denom) if denom > 0 else 0.0
                    elif obj == 'ap':
                        p_arr, r_arr, th_arr = precision_recall_curve(yti, ypi)
                        if th_arr is not None and len(th_arr) > 0:
                            d = (1.0 - p_arr[1:])**2 + (1.0 - r_arr[1:])**2
                            idx = int(np.argmin(d))
                            score_obj = 1.0 - float(d[idx])
                            best_th = float(th_arr[idx])
                            thresholds[i] = best_th
                            break
                        else:
                            score_obj = 0.0
                    elif obj == 'ap_f1':
                        p_arr, r_arr, th_arr = precision_recall_curve(yti, ypi)
                        if th_arr is not None and len(th_arr) > 0:
                            p_use = p_arr[1:]
                            r_use = r_arr[1:]
                            denom = (p_use + r_use)
                            f1_arr = (2.0 * p_use * r_use / denom)
                            f1_arr = np.where(denom > 0, f1_arr, 0.0)
                            idx = int(np.argmax(f1_arr))
                            score_obj = float(f1_arr[idx])
                            best_th = float(th_arr[idx])
                            thresholds[i] = best_th
                            break
                        else:
                            score_obj = 0.0
                    else:
                        score_obj = f1_score(yti, yb)
                except Exception:
                    score_obj = 0.0
                if score_obj > best_score:
                    best_score = score_obj
                    best_th = th
            thresholds[i] = best_th
        # 保存并缓存
        self.per_target_thresholds = thresholds
        try:
            import pandas as pd
            names = target_names if target_names and len(target_names) == num_targets else [f"lr_{i}" for i in range(num_targets)]
            df = pd.DataFrame({"lr": names, "threshold": thresholds})
            df.to_csv(epoch_dir / "per_target_thresholds.csv", index=False)
        except Exception:
            pass

    def _save_per_lr_plots(self, epoch: int, y_pred: torch.Tensor, y_true: torch.Tensor, target_names: Optional[List[str]]) -> None:
        epoch_dir = self._ensure_epoch_dir(epoch)
        plots_dir = epoch_dir / 'lr_plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(y_pred, torch.Tensor):
            yp = y_pred.detach().cpu().numpy()
        else:
            yp = y_pred
        if isinstance(y_true, torch.Tensor):
            yt = y_true.detach().cpu().numpy()
        else:
            yt = y_true
        num_targets = yp.shape[1]
        max_plots = int(self.config.get('evaluation', {}).get('max_lr_plots', num_targets))
        count = min(max_plots, num_targets)
        for i in range(count):
            fig = plt.figure(figsize=(5, 4))
            ax = fig.add_subplot(111)
            name = target_names[i] if target_names and i < len(target_names) else f'lr_{i}'
            if self.task_type == 'classification':
                ax.scatter(yt[:, i], yp[:, i], alpha=0.6, s=2)
                ax.axhline(y=self.pred_threshold, color='r', linestyle='--', alpha=0.6)
                ax.set_xlabel('True')
                ax.set_ylabel('Prob')
                ax.set_title(str(name))
            else:
                ax.scatter(yt[:, i], yp[:, i], alpha=0.6, s=2)
                min_val = float(min(yt[:, i].min(), yp[:, i].min()))
                max_val = float(max(yt[:, i].max(), yp[:, i].max()))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.6)
                ax.set_xlabel('True')
                ax.set_ylabel('Pred')
                ax.set_title(str(name))
            ax.grid(True, alpha=0.3)
            safe_name = str(name).replace(' ', '_').replace('/', '_')
            out_path = plots_dir / f"{i}_{safe_name}.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)

    def _save_per_lr_classification_visuals(self, epoch: int, y_pred: torch.Tensor, y_true: torch.Tensor, target_names: Optional[List[str]]) -> None:
        epoch_dir = self._ensure_epoch_dir(epoch)
        base_dir = epoch_dir / 'lr_classification'
        base_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(y_pred, torch.Tensor):
            yp = y_pred.detach().cpu().numpy()
        else:
            yp = y_pred
        if isinstance(y_true, torch.Tensor):
            yt = y_true.detach().cpu().numpy()
        else:
            yt = y_true
        import numpy as np
        try:
            from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, average_precision_score
        except Exception:
            return
        num_targets = yp.shape[1]
        rank_metric = str(self.config.get('evaluation', {}).get('visual_rank_metric', 'auc')).lower()
        top_k_cfg = int(self.config.get('evaluation', {}).get('top_k_lr_visuals', 50))
        min_pos_ratio = float(self.config.get('evaluation', {}).get('min_pos_ratio_for_visuals', 0.0))
        ranked = []
        for i in range(num_targets):
            pos_count = int(np.sum(yt[:, i] == 1))
            neg_count = int(np.sum(yt[:, i] == 0))
            total_count = pos_count + neg_count
            pos_ratio = (pos_count / total_count) if total_count > 0 else 0.0
            if (min_pos_ratio > 0.0 and pos_ratio < min_pos_ratio) or neg_count == 0:
                continue
            if rank_metric == 'ap':
                try:
                    score_i = average_precision_score(yt[:, i], yp[:, i])
                except Exception:
                    score_i = -1.0
            else:
                try:
                    fpr_i, tpr_i, _ = roc_curve(yt[:, i], yp[:, i])
                    score_i = auc(fpr_i, tpr_i)
                except Exception:
                    score_i = -1.0
            ranked.append((i, float(score_i) if score_i is not None else -1.0))
        ranked = [kv for kv in ranked if kv[1] >= 0.0]
        ranked.sort(key=lambda x: x[1], reverse=True)
        top_k = min(top_k_cfg, len(ranked))
        indices = [idx for idx, _ in ranked[:top_k]]
        for i in indices:
            if np.unique(yt[:, i]).size < 2:
                continue
            name = target_names[i] if target_names and i < len(target_names) else f'lr_{i}'
            safe_name = str(name).replace(' ', '_').replace('/', '_')
            ypi = yp[:, i]
            yti = yt[:, i]
            th_i = float(self.pred_threshold)
            if hasattr(self, 'per_target_thresholds') and i < len(self.per_target_thresholds):
                th_i = float(self.per_target_thresholds[i])
            yb = (ypi >= th_i).astype(int)
            try:
                cm = confusion_matrix(yti, yb)
                acc = accuracy_score(yti, yb)
                prec = precision_score(yti, yb)
                rec = recall_score(yti, yb)
                f1 = f1_score(yti, yb)
            except Exception:
                continue
            try:
                fpr, tpr, _ = roc_curve(yti, ypi)
                roc_auc = auc(fpr, tpr)
            except Exception:
                fpr, tpr, roc_auc = None, None, 0.0
            fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(str(name))
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            plt.tight_layout()
            plt.savefig(base_dir / f'cm_{i}_{safe_name}.png', dpi=200, bbox_inches='tight')
            plt.close(fig)
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            metrics_vals = [acc, f1, prec, rec]
            ax.bar(['accuracy', 'f1', 'precision', 'recall'], metrics_vals)
            ax.set_ylim(0.0, 1.0)
            ax.set_title(str(name))
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(base_dir / f'metrics_{i}_{safe_name}.png', dpi=200, bbox_inches='tight')
            plt.close(fig)
            if fpr is not None and tpr is not None:
                fig, ax = plt.subplots(1, 1, figsize=(5, 4))
                ax.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
                ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
                ax.set_title(str(name))
                ax.set_xlabel('FPR')
                ax.set_ylabel('TPR')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(base_dir / f'roc_{i}_{safe_name}.png', dpi=200, bbox_inches='tight')
                plt.close(fig)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.is_data_parallel:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        logger.info(f"已加载检查点: {checkpoint_path}")
        logger.info(f"Epoch: {checkpoint['epoch']}, 验证损失: {checkpoint['val_loss']:.4f}")
        
        return checkpoint['epoch'], checkpoint['val_loss'], checkpoint['metrics']
        # 分类任务：根据训练集不平衡动态设置正类权重
        if self.task_type == 'classification' and bool(self.config.get('training', {}).get('use_pos_weight_loss', False)):
            try:
                pos_weight = self._compute_pos_weight(train_loader)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
                logger.info("已启用按目标的正类加权损失")
            except Exception as e:
                logger.warning(f"计算正类权重失败，继续使用默认损失: {e}")
            # 分类任务：为每个目标调优阈值以提升macro指标与可视化质量
            if self.task_type == 'classification':
                try:
                    self._compute_and_save_per_target_thresholds(epoch, val_pred, val_true, self.metadata.get('lr_pairs'))
                except Exception as e:
                    logger.warning(f"逐目标阈值调优失败: {e}")
