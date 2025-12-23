#!/usr/bin/env python3
"""
GNN配体受体预测模型 - 主训练脚本
"""
import os
import sys
import argparse
import logging
import yaml
import torch
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from src.data.preprocessing import DataPreprocessor
from src.models.gnn_model import create_model
from src.training.trainer import GNNTrainer
from src.training.hyperopt import HyperparameterOptimizer
from src.training.cross_validation import CrossValidator
from src.evaluation.evaluator import ModelEvaluator
from src.utils.gpu_utils import GPUManager

def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log', encoding='utf-8')
        ]
    )

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='GNN配体受体预测模型训练')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['train', 'hyperopt', 'evaluate', 'cv'],
                       default='train', help='运行模式')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='检查点路径（用于恢复训练）')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    parser.add_argument('--conda-env', type=str, default='deeplearning',
                       help='Conda环境名称')
    parser.add_argument('--test', action='store_true',
                       help='使用测试数据集（/home/lilj/work/xenium/testdata 或 /home/lilj/work/xenium/testdata/cell）')
    # 分类与数据目录/H5支持
    parser.add_argument('--classification', action='store_true',
                       help='强制使用分类任务（使用padj标签）')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='覆盖配置中的数据目录，例如 /home/lilj/work/xenium/data/cell')
    parser.add_argument('--file-format', type=str, choices=['csv', 'h5'], default=None,
                       help='数据文件格式：csv 或 h5。h5时读取 x.h5/padj.h5 或 y.h5')
    parser.add_argument('--x-key', type=str, default=None,
                       help='H5中x数据的key，例如 X 或 x')
    parser.add_argument('--y-key', type=str, default=None,
                       help='H5中y数据的key，例如 Y 或 y（回归标签）')
    parser.add_argument('--padj-key', type=str, default=None,
                       help='H5中padj数据的key，例如 padj 或 Y（分类标签）')
    parser.add_argument('--cv-type', type=str, choices=['kfold', 'stratified'],
                       default='kfold', help='交叉验证类型')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='交叉验证折数')
    parser.add_argument('--save-cv-models', action='store_true',
                       help='保存每折的模型')
    
    # 阈值和损失控制参数
    parser.add_argument('--no-dynamic-threshold', action='store_true',
                       help='关闭逐目标动态阈值计算，使用统一阈值')
    parser.add_argument('--unified-threshold', type=float, default=None,
                       help='统一阈值设定 (覆盖配置中的pred_threshold，默认0.5)')
    parser.add_argument('--no-pos-weight', action='store_true',
                       help='关闭BCEWithLogitsLoss中的pos_weight正类加权')

    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("GNN配体受体预测模型训练开始")
    logger.info("=" * 60)
    
    # 检查conda环境
    current_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
    if current_env != args.conda_env:
        logger.warning(f"当前conda环境: {current_env}, 期望环境: {args.conda_env}")
        logger.warning("请确保在正确的conda环境中运行")
    
    try:
        # 加载配置
        logger.info(f"加载配置文件: {args.config}")
        config = load_config(args.config)
        
        # 检查GPU
        gpu_manager = GPUManager()
        available_gpus = gpu_manager.detect_gpus()
        logger.info(f"检测到{len(available_gpus)}个可用GPU: {available_gpus}")
        
        # 数据预处理
        logger.info("开始数据预处理...")

        # 根据命令行覆盖数据目录/任务类型/文件格式
        if args.data_dir:
            config.setdefault('data', {})
            config['data']['data_dir'] = args.data_dir
            logger.info(f"命令行覆盖数据目录: {args.data_dir}")
        if args.file_format:
            config.setdefault('data', {})
            config['data']['file_format'] = args.file_format
            logger.info(f"命令行设置文件格式: {args.file_format}")
        # H5键名覆盖
        if args.x_key or args.y_key or args.padj_key:
            config.setdefault('data', {})
            config['data'].setdefault('h5_keys', {})
            if args.x_key:
                config['data']['h5_keys']['x'] = args.x_key
            if args.y_key:
                config['data']['h5_keys']['y'] = args.y_key
            if args.padj_key:
                config['data']['h5_keys']['padj'] = args.padj_key
            logger.info(f"H5键名覆盖: {config['data']['h5_keys']}")

        # 阈值和权重控制覆盖
        config.setdefault('classification', {})
        if args.no_dynamic_threshold:
            config['classification']['use_dynamic_threshold'] = False
            logger.info("命令行: 关闭逐目标动态阈值计算")
        
        if args.unified_threshold is not None:
            config['classification']['pred_threshold'] = args.unified_threshold
            logger.info(f"命令行: 设置统一阈值 = {args.unified_threshold}")
            
        if args.no_pos_weight:
            config['classification']['use_pos_weight_loss'] = False
            logger.info("命令行: 关闭正类加权 (pos_weight)")

        # 如果使用测试数据，修改配置中的数据路径
        if args.test:
            # 优先选择 cell 目录
            cell_dir = '/home/lilj/work/xenium/testdata/cell'
            default_dir = '/home/lilj/work/xenium/testdata'
            if os.path.exists(os.path.join(cell_dir, 'x.csv')):
                logger.info(f"使用测试数据集: {cell_dir}")
                config['data']['data_dir'] = cell_dir
                # 若存在padj.csv且任务为分类，则仅设置路径，不更改任务类型
                padj_path = os.path.join(cell_dir, 'padj.csv')
                if os.path.exists(padj_path) and config.get('task_type') == 'classification':
                    config.setdefault('classification', {})
                    config['data']['padj_path'] = padj_path
                    logger.info(f"检测到padj标签文件: {padj_path}（保持task_type={config.get('task_type')}）")
            else:
                logger.info(f"使用测试数据集: {default_dir}")
                config['data']['data_dir'] = default_dir

        # 根据任务类型统一设置评估指标列表（保持训练/验证/测试一致）
        task_type = config.get('task_type', 'regression')
        if task_type == 'classification':
            config['evaluation']['metrics'] = [
                'accuracy_micro', 'accuracy_macro',
                'f1_micro', 'f1_macro',
                'roc_auc_micro', 'roc_auc_macro',
                'precision_micro', 'precision_macro',
                'recall_micro', 'recall_macro'
            ]
        else:
            config['evaluation']['metrics'] = [
                'mse', 'mae', 'r2', 'pearson'
            ]
        
        # 测试模式下，降低batch_size以避免OOM，并优先使用检测到的GPU列表
        try:
            if args.test:
                orig_bs = int(config.get('training', {}).get('batch_size', 64))
                safe_bs = min(orig_bs, 4)
                config.setdefault('training', {})['batch_size'] = safe_bs
                if available_gpus:
                    config.setdefault('gpu', {})['device_ids'] = available_gpus
                logger.info(f"测试模式: 将batch_size从{orig_bs}调整为{safe_bs}，device_ids={config['gpu'].get('device_ids')}")
        except Exception:
            pass
        
        preprocessor = DataPreprocessor(config)
        train_loader, val_loader, test_loader, metadata = preprocessor.process_all()
        
        logger.info(f"数据预处理完成:")
        logger.info(f"  基因数量: {metadata['num_genes']}")
        logger.info(f"  配体受体对数量: {metadata['num_lr_pairs']}")
        logger.info(f"  通路数量: {metadata['num_pathways']}")
        logger.info(f"  训练样本: {len(train_loader.dataset)}")
        logger.info(f"  验证样本: {len(val_loader.dataset)}")
        logger.info(f"  测试样本: {len(test_loader.dataset)}")
        
        if args.mode == 'hyperopt':
            # 超参数优化模式
            logger.info("开始超参数优化...")
            optimizer = HyperparameterOptimizer(config, metadata, train_loader, val_loader)
            study = optimizer.optimize()
            
            # 获取最佳配置
            best_config = optimizer.get_best_config(study)
            
            # 保存最佳配置
            best_config_path = Path(config['output']['result_dir']) / 'best_config.yaml'
            with open(best_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(best_config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"最佳配置已保存到: {best_config_path}")
            logger.info("超参数优化完成")
            
        elif args.mode == 'cv':
            # 交叉验证模式
            logger.info(f"开始{args.cv_folds}折交叉验证 (类型: {args.cv_type})...")
            
            # 创建交叉验证器
            cv = CrossValidator(config, metadata)
            
            # 合并训练和验证数据用于交叉验证
            from torch.utils.data import ConcatDataset
            cv_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
            
            # 运行交叉验证
            cv_results = cv.run_cross_validation(
                dataset=cv_dataset,
                test_dataset=test_loader.dataset,
                cv_type=args.cv_type,
                n_splits=args.cv_folds,
                save_models=args.save_cv_models
            )
            
            logger.info("交叉验证完成!")
            logger.info(f"平均验证 R²: {cv_results['val_metrics_mean']['r2']:.4f} ± {cv_results['val_metrics_std']['r2']:.4f}")
            if cv_results['test_metrics_mean']:
                logger.info(f"平均测试 R²: {cv_results['test_metrics_mean']['r2']:.4f} ± {cv_results['test_metrics_std']['r2']:.4f}")
            
        elif args.mode == 'train':
            # 训练模式
            logger.info("开始模型训练...")
            
            # 创建模型
            model = create_model(config, metadata)
            
            # 创建训练器
            trainer = GNNTrainer(model, config, metadata)
            
            # 恢复检查点（如果指定）
            start_epoch = 0
            if args.checkpoint:
                logger.info(f"从检查点恢复训练: {args.checkpoint}")
                start_epoch, _, _ = trainer.load_checkpoint(args.checkpoint)
            
            # 训练模型
            results = trainer.train(train_loader, val_loader, start_epoch)
            
            logger.info("训练完成!")
            logger.info(f"最终训练损失: {results['final_train_loss']:.4f}")
            logger.info(f"最终验证损失: {results['final_val_loss']:.4f}")
            logger.info(f"最佳验证损失: {results['best_val_loss']:.4f}")
            logger.info(f"训练轮数: {results['total_epochs']}")
            logger.info(f"训练时间: {results['total_time']/3600:.2f}小时")
            
            # 在测试集上评估
            logger.info("在测试集上评估模型...")
            model.eval()
            evaluator = ModelEvaluator(config['evaluation']['metrics'])
            if getattr(trainer, 'per_target_thresholds', None) is not None:
                import numpy as np
                evaluator.thresholds = np.array(trainer.per_target_thresholds)
            
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch in test_loader:
                    batch = trainer._move_batch_to_device(batch)
                    predictions = model(batch)
                    # 分类任务：使用概率
                    if config.get('task_type', 'regression') == 'classification':
                        predictions = torch.sigmoid(predictions)
                    all_predictions.append(predictions.cpu())
                    all_targets.append(batch['lr_scores'].cpu())
            
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # 生成评估报告
            test_report = evaluator.generate_report(
                all_predictions, all_targets,
                target_names=metadata['lr_pairs'],
                save_dir=Path(config['output']['result_dir']) / 'test_evaluation'
            )
            
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in test_report['overall_metrics'].items()])
            logger.info(f"测试指标 - {metrics_str}")
            
        elif args.mode == 'evaluate':
            # 评估模式
            logger.info("评估模式...")
            
            # 加载最佳模型
            model_path = Path(config['output']['model_dir']) / 'best_model.pth'
            if not model_path.exists():
                raise FileNotFoundError(f"找不到模型文件: {model_path}")
            
            checkpoint = torch.load(model_path, map_location='cpu')
            model = create_model(checkpoint['config'], checkpoint['metadata'])
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 设置设备
            gpu_manager = GPUManager()
            device = gpu_manager.setup_device(
                device_ids=config.get('gpu', {}).get('device_ids', list(range(torch.cuda.device_count()))),
                min_memory_gb=2.0
            )
            model.to(device)
            is_data_parallel = False
            if device.type == 'cuda':
                try:
                    gpu_count = torch.cuda.device_count()
                    if gpu_count > 1:
                        device_ids = config.get('gpu', {}).get('device_ids', list(range(gpu_count)))
                        if not isinstance(device_ids, list) or len(device_ids) == 0:
                            device_ids = list(range(gpu_count))
                        model = torch.nn.DataParallel(model, device_ids=device_ids)
                        is_data_parallel = True
                        logger.info(f"评估启用DataParallel，使用GPU: {device_ids}")
                except Exception as e:
                    logger.warning(f"评估启用DataParallel失败，回退到单GPU: {e}")
            model.eval()
            
            logger.info(f"已加载模型: {model_path}")
            
            # 在测试集上评估
            evaluator = ModelEvaluator(config['evaluation']['metrics'])
            
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch in test_loader:
                    eval_batch = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            if is_data_parallel:
                                eval_batch[k] = v
                            else:
                                eval_batch[k] = v.to(device, non_blocking=True)
                        else:
                            eval_batch[k] = v
                    batch = eval_batch
                    predictions = model(batch)
                    if config.get('task_type', 'regression') == 'classification':
                        predictions = torch.sigmoid(predictions)
                    all_predictions.append(predictions.cpu())
                    all_targets.append(batch['lr_scores'].cpu())
            
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # 生成详细评估报告
            eval_report = evaluator.generate_report(
                all_predictions, all_targets,
                target_names=metadata['lr_pairs'],
                save_dir=Path(config['output']['result_dir']) / 'final_evaluation'
            )
            
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in eval_report['overall_metrics'].items()])
            logger.info(f"测试指标 - {metrics_str}")
        
        logger.info("=" * 60)
        logger.info("程序执行完成")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()
