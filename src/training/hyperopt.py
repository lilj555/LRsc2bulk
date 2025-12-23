"""
超参数优化模块
"""
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..models.gnn_model import create_model
from ..training.trainer import GNNTrainer
from ..utils.gpu_utils import GPUManager

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self, config: dict, metadata: dict, 
                 train_loader: DataLoader, val_loader: DataLoader):
        self.config = config
        self.metadata = metadata
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # GPU管理
        self.gpu_manager = GPUManager()
        
        # 优化配置
        self.hyperopt_config = config['hyperopt']
        self.n_trials = self.hyperopt_config['n_trials']
        self.timeout = self.hyperopt_config.get('timeout', 3600)
        self.search_space = self.hyperopt_config['search_space']
        
        # 结果保存
        self.output_dir = Path(config['output']['result_dir']) / 'hyperopt'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 最佳结果跟踪
        self.best_trial = None
        self.best_value = float('inf')
        
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """建议超参数"""
        suggested_params = {}
        
        for param_name, param_config in self.search_space.items():
            if isinstance(param_config, list):
                # 分类参数
                suggested_params[param_name] = trial.suggest_categorical(
                    param_name, param_config
                )
            elif isinstance(param_config, tuple) and len(param_config) == 2:
                # 连续参数
                if isinstance(param_config[0], int) and isinstance(param_config[1], int):
                    # 整数参数
                    suggested_params[param_name] = trial.suggest_int(
                        param_name, param_config[0], param_config[1]
                    )
                else:
                    # 浮点参数
                    suggested_params[param_name] = trial.suggest_float(
                        param_name, param_config[0], param_config[1], log=True
                    )
            else:
                logger.warning(f"不支持的参数配置格式: {param_name}: {param_config}")
                
        return suggested_params
    
    def update_config_with_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """用建议的参数更新配置"""
        updated_config = self.config.copy()
        
        # 更新模型参数
        if 'gene_embedding_dim' in params:
            updated_config['model']['gene_embedding_dim'] = params['gene_embedding_dim']
        if 'hidden_dim' in params:
            updated_config['model']['hidden_dim'] = params['hidden_dim']
        if 'gat_layers' in params:
            updated_config['model']['gat_layers'] = params['gat_layers']
        if 'attention_heads' in params:
            updated_config['model']['attention_heads'] = params['attention_heads']
        if 'dropout' in params:
            updated_config['model']['dropout'] = params['dropout']
        
        # 更新训练参数
        if 'learning_rate' in params:
            updated_config['training']['learning_rate'] = params['learning_rate']
        if 'batch_size' in params:
            updated_config['training']['batch_size'] = params['batch_size']
        if 'weight_decay' in params:
            updated_config['training']['weight_decay'] = params['weight_decay']
        
        # 为超参数优化调整训练设置
        updated_config['training']['epochs'] = min(50, updated_config['training']['epochs'])
        updated_config['training']['patience'] = min(10, updated_config['training']['patience'])
        updated_config['output']['checkpoint_freq'] = 999  # 不保存检查点
        
        return updated_config
    
    def objective(self, trial: optuna.Trial) -> float:
        """优化目标函数"""
        try:
            # 建议超参数
            params = self.suggest_hyperparameters(trial)
            logger.info(f"Trial {trial.number}: 测试参数 {params}")
            
            # 更新配置
            trial_config = self.update_config_with_params(params)
            
            # 创建模型
            model = create_model(trial_config, self.metadata)
            
            # 创建训练器
            trainer = GNNTrainer(model, trial_config, self.metadata)
            
            # 如果批次大小改变，需要重新创建数据加载器
            if params.get('batch_size', self.config['training']['batch_size']) != self.config['training']['batch_size']:
                from torch.utils.data import DataLoader
                
                train_dataset = self.train_loader.dataset
                val_dataset = self.val_loader.dataset
                
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=params['batch_size'],
                    shuffle=True,
                    num_workers=self.config['gpu']['num_workers'],
                    pin_memory=True,
                    drop_last=True
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=params['batch_size'],
                    shuffle=False,
                    num_workers=self.config['gpu']['num_workers'],
                    pin_memory=True
                )
            else:
                train_loader = self.train_loader
                val_loader = self.val_loader
            
            # 训练模型
            results = trainer.train(train_loader, val_loader)
            
            # 返回验证损失作为优化目标
            objective_value = results['best_val_loss']
            
            # 记录额外信息
            trial.set_user_attr('final_train_loss', results['final_train_loss'])
            trial.set_user_attr('final_val_loss', results['final_val_loss'])
            trial.set_user_attr('total_epochs', results['total_epochs'])
            trial.set_user_attr('total_time', results['total_time'])
            
            if 'final_metrics' in results:
                for metric_name, metric_value in results['final_metrics'].items():
                    trial.set_user_attr(f'final_{metric_name}', metric_value)
            
            logger.info(f"Trial {trial.number} 完成: 验证损失 = {objective_value:.4f}")
            
            # 清理GPU内存
            del model, trainer
            torch.cuda.empty_cache()
            
            return objective_value
            
        except Exception as e:
            logger.error(f"Trial {trial.number} 失败: {e}")
            # 清理GPU内存
            torch.cuda.empty_cache()
            return float('inf')
    
    def optimize(self, study_name: str = "gnn_hyperopt") -> optuna.Study:
        """执行超参数优化"""
        logger.info("开始超参数优化...")
        logger.info(f"试验次数: {self.n_trials}")
        logger.info(f"超时时间: {self.timeout}秒")
        logger.info(f"搜索空间: {self.search_space}")
        
        # 创建研究
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=5
            )
        )
        
        # 执行优化
        start_time = time.time()
        
        try:
            study.optimize(
                self.objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                callbacks=[self._trial_callback]
            )
        except KeyboardInterrupt:
            logger.info("优化被用户中断")
        
        total_time = time.time() - start_time
        
        # 保存结果
        self._save_results(study, total_time)
        
        # 生成报告
        self._generate_report(study)
        
        logger.info(f"超参数优化完成，总时间: {total_time/3600:.2f}小时")
        logger.info(f"最佳验证损失: {study.best_value:.4f}")
        logger.info(f"最佳参数: {study.best_params}")
        
        return study
    
    def _trial_callback(self, study: optuna.Study, trial: optuna.Trial):
        """试验回调函数"""
        if trial.value < self.best_value:
            self.best_value = trial.value
            self.best_trial = trial
            logger.info(f"发现新的最佳试验 {trial.number}: {trial.value:.4f}")
    
    def _save_results(self, study: optuna.Study, total_time: float):
        """保存优化结果"""
        # 保存研究对象
        study_path = self.output_dir / 'study.pkl'
        import pickle
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        
        # 保存最佳参数
        best_params_path = self.output_dir / 'best_params.json'
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        # 保存试验历史
        trials_df = study.trials_dataframe()
        trials_df.to_csv(self.output_dir / 'trials_history.csv', index=False)
        
        # 保存优化摘要
        summary = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'total_time_hours': total_time / 3600,
            'best_trial_number': study.best_trial.number
        }
        
        if study.best_trial.user_attrs:
            summary['best_trial_metrics'] = study.best_trial.user_attrs
        
        summary_path = self.output_dir / 'optimization_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"优化结果已保存到: {self.output_dir}")
    
    def _generate_report(self, study: optuna.Study):
        """生成优化报告"""
        # 优化历史图
        self._plot_optimization_history(study)
        
        # 参数重要性图
        self._plot_param_importance(study)
        
        # 参数关系图
        self._plot_param_relationships(study)
        
        # 试验分布图
        self._plot_trial_distribution(study)
    
    def _plot_optimization_history(self, study: optuna.Study):
        """绘制优化历史"""
        fig = go.Figure()
        
        trials = study.trials
        trial_numbers = [t.number for t in trials if t.value is not None]
        values = [t.value for t in trials if t.value is not None]
        
        # 累积最佳值
        best_values = []
        current_best = float('inf')
        for value in values:
            if value < current_best:
                current_best = value
            best_values.append(current_best)
        
        # 所有试验值
        fig.add_trace(go.Scatter(
            x=trial_numbers,
            y=values,
            mode='markers',
            name='试验值',
            marker=dict(color='lightblue', size=6)
        ))
        
        # 最佳值曲线
        fig.add_trace(go.Scatter(
            x=trial_numbers,
            y=best_values,
            mode='lines',
            name='最佳值',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='超参数优化历史',
            xaxis_title='试验编号',
            yaxis_title='验证损失',
            hovermode='x unified'
        )
        
        fig.write_html(self.output_dir / 'optimization_history.html')
        
        # 保存静态图
        fig.write_image(self.output_dir / 'optimization_history.png')
    
    def _plot_param_importance(self, study: optuna.Study):
        """绘制参数重要性"""
        try:
            importance = optuna.importance.get_param_importances(study)
            
            if importance:
                params = list(importance.keys())
                values = list(importance.values())
                
                fig = go.Figure(data=[
                    go.Bar(x=values, y=params, orientation='h')
                ])
                
                fig.update_layout(
                    title='参数重要性',
                    xaxis_title='重要性',
                    yaxis_title='参数',
                    height=max(400, len(params) * 30)
                )
                
                fig.write_html(self.output_dir / 'param_importance.html')
                fig.write_image(self.output_dir / 'param_importance.png')
        except Exception as e:
            logger.warning(f"无法生成参数重要性图: {e}")
    
    def _plot_param_relationships(self, study: optuna.Study):
        """绘制参数关系"""
        try:
            trials_df = study.trials_dataframe()
            param_cols = [col for col in trials_df.columns if col.startswith('params_')]
            
            if len(param_cols) >= 2:
                # 选择前几个重要参数
                param_cols = param_cols[:min(4, len(param_cols))]
                
                fig = make_subplots(
                    rows=len(param_cols), cols=len(param_cols),
                    subplot_titles=[col.replace('params_', '') for col in param_cols]
                )
                
                for i, param1 in enumerate(param_cols):
                    for j, param2 in enumerate(param_cols):
                        if i != j:
                            fig.add_trace(
                                go.Scatter(
                                    x=trials_df[param1],
                                    y=trials_df[param2],
                                    mode='markers',
                                    marker=dict(
                                        color=trials_df['value'],
                                        colorscale='Viridis',
                                        showscale=(i == 0 and j == 1)
                                    ),
                                    showlegend=False
                                ),
                                row=i+1, col=j+1
                            )
                
                fig.update_layout(
                    title='参数关系图',
                    height=200 * len(param_cols)
                )
                
                fig.write_html(self.output_dir / 'param_relationships.html')
        except Exception as e:
            logger.warning(f"无法生成参数关系图: {e}")
    
    def _plot_trial_distribution(self, study: optuna.Study):
        """绘制试验分布"""
        trials_df = study.trials_dataframe()
        
        if 'value' in trials_df.columns:
            fig = go.Figure(data=[
                go.Histogram(x=trials_df['value'], nbinsx=30)
            ])
            
            fig.update_layout(
                title='试验值分布',
                xaxis_title='验证损失',
                yaxis_title='频数'
            )
            
            fig.write_html(self.output_dir / 'trial_distribution.html')
            fig.write_image(self.output_dir / 'trial_distribution.png')
    
    def get_best_config(self, study: optuna.Study) -> Dict[str, Any]:
        """获取最佳配置"""
        best_params = study.best_params
        best_config = self.update_config_with_params(best_params)
        
        # 恢复完整训练设置
        best_config['training']['epochs'] = self.config['training']['epochs']
        best_config['training']['patience'] = self.config['training']['patience']
        best_config['output']['checkpoint_freq'] = self.config['output']['checkpoint_freq']
        
        return best_config