#!/usr/bin/env python3
"""
创建测试数据集 - 从原始数据中随机抽取1万个样本
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
import logging

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_test_dataset(data_dir: str, test_dir: str, n_samples: int = 10000, random_seed: int = 42):
    """
    从原始数据中随机抽取样本创建测试数据集
    
    Args:
        data_dir: 原始数据目录
        test_dir: 测试数据目录
        n_samples: 抽取样本数量
        random_seed: 随机种子
    """
    logger = logging.getLogger(__name__)
    
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 创建测试数据目录
    test_path = Path(test_dir)
    test_path.mkdir(parents=True, exist_ok=True)
    
    # 文件路径
    x_path = Path(data_dir) / "x.csv"
    y_path = Path(data_dir) / "y.csv"
    pathway_path = Path(data_dir) / "pathway_gene.csv"
    
    logger.info(f"开始从 {data_dir} 抽取 {n_samples} 个样本到 {test_dir}")
    
    # 检查文件是否存在
    if not x_path.exists():
        raise FileNotFoundError(f"基因表达文件不存在: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"配体受体文件不存在: {y_path}")
    if not pathway_path.exists():
        raise FileNotFoundError(f"通路基因文件不存在: {pathway_path}")
    
    # 读取数据
    logger.info("读取基因表达数据...")
    x_df = pd.read_csv(x_path, index_col=0)
    logger.info(f"基因表达数据形状: {x_df.shape}")
    
    logger.info("读取配体受体数据...")
    y_df = pd.read_csv(y_path, index_col=0)
    logger.info(f"配体受体数据形状: {y_df.shape}")
    
    # 检查样本名是否对应
    if not x_df.index.equals(y_df.index):
        logger.warning("x和y的样本索引不完全匹配，将使用交集")
        common_samples = x_df.index.intersection(y_df.index)
        x_df = x_df.loc[common_samples]
        y_df = y_df.loc[common_samples]
        logger.info(f"共同样本数量: {len(common_samples)}")
    
    # 检查样本数量
    total_samples = len(x_df)
    if n_samples > total_samples:
        logger.warning(f"请求样本数 ({n_samples}) 大于总样本数 ({total_samples})，将使用全部样本")
        n_samples = total_samples
    
    # 随机抽取样本
    logger.info(f"随机抽取 {n_samples} 个样本...")
    sample_indices = np.random.choice(x_df.index, size=n_samples, replace=False)
    
    # 提取对应的数据
    x_test = x_df.loc[sample_indices]
    y_test = y_df.loc[sample_indices]
    
    # 保存测试数据
    x_test_path = test_path / "x.csv"
    y_test_path = test_path / "y.csv"
    pathway_test_path = test_path / "pathway_gene.csv"
    
    logger.info(f"保存测试数据到 {x_test_path}")
    x_test.to_csv(x_test_path)
    
    logger.info(f"保存测试数据到 {y_test_path}")
    y_test.to_csv(y_test_path)
    
    # 复制通路基因文件（这个文件不需要抽样）
    logger.info(f"复制通路基因文件到 {pathway_test_path}")
    pathway_df = pd.read_csv(pathway_path)
    pathway_df.to_csv(pathway_test_path, index=False)
    
    # 输出统计信息
    logger.info("=" * 50)
    logger.info("测试数据集创建完成!")
    logger.info(f"测试数据目录: {test_dir}")
    logger.info(f"样本数量: {n_samples}")
    logger.info(f"基因数量: {x_test.shape[1]}")
    logger.info(f"配体受体对数量: {y_test.shape[1]}")
    logger.info(f"通路基因映射: {len(pathway_df)} 条记录")
    logger.info("=" * 50)
    
    return {
        'n_samples': n_samples,
        'n_genes': x_test.shape[1],
        'n_lr_pairs': y_test.shape[1],
        'n_pathways': len(pathway_df['pathway_id'].unique()) if 'pathway_id' in pathway_df.columns else 0
    }

def main():
    parser = argparse.ArgumentParser(description='创建测试数据集')
    parser.add_argument('--data-dir', type=str, 
                       default='/home/lilj/work/xenium/data',
                       help='原始数据目录')
    parser.add_argument('--test-dir', type=str,
                       default='/home/lilj/work/xenium/testdata',
                       help='测试数据目录')
    parser.add_argument('--n-samples', type=int, default=10000,
                       help='抽取样本数量')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        stats = create_test_dataset(
            data_dir=args.data_dir,
            test_dir=args.test_dir,
            n_samples=args.n_samples,
            random_seed=args.random_seed
        )
        print(f"测试数据集创建成功: {stats}")
        
    except Exception as e:
        logging.error(f"创建测试数据集失败: {e}")
        raise

if __name__ == "__main__":
    main()