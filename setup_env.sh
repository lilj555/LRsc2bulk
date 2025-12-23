#!/bin/bash
# GNN配体受体预测模型环境设置脚本

echo "=========================================="
echo "GNN配体受体预测模型环境设置"
echo "=========================================="

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到conda，请先安装Anaconda或Miniconda"
    exit 1
fi

# 激活deeplearning环境
echo "激活conda环境: deeplearning"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deeplearning

# 检查环境是否存在
if [ "$CONDA_DEFAULT_ENV" != "deeplearning" ]; then
    echo "警告: deeplearning环境不存在或激活失败"
    echo "请确保已创建deeplearning环境"
    exit 1
fi

echo "当前conda环境: $CONDA_DEFAULT_ENV"

# 检查Python版本
python_version=$(python --version 2>&1)
echo "Python版本: $python_version"

# 检查CUDA是否可用
echo "检查CUDA可用性..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU数量: {torch.cuda.device_count()}')"

# 安装依赖（如果需要）
echo "检查并安装依赖包..."

# 检查关键包是否已安装
packages_to_check=("torch" "torch_geometric" "pandas" "numpy" "scikit-learn" "optuna" "matplotlib" "seaborn" "pyyaml")

for package in "${packages_to_check[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo "✓ $package 已安装"
    else
        echo "✗ $package 未安装"
        echo "请运行以下命令安装依赖:"
        echo "pip install -r requirements.txt"
        exit 1
    fi
done

# 检查GPU内存
echo "检查GPU状态..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
else
    echo "nvidia-smi不可用，无法检查GPU状态"
fi

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES="0"  # 默认使用第一个GPU

echo "=========================================="
echo "环境设置完成!"
echo "=========================================="
echo "使用方法:"
echo "1. 训练模型:"
echo "   python train.py --mode train"
echo ""
echo "2. 超参数优化:"
echo "   python train.py --mode hyperopt"
echo ""
echo "3. 评估模型:"
echo "   python train.py --mode evaluate"
echo ""
echo "4. 查看帮助:"
echo "   python train.py --help"
echo "=========================================="