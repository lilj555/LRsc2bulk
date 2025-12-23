import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# 使用用户提供的路径
csv_path = "/home/lilj/work/xenium/tcga_output/LR_survival_analysis_20251222_194501.csv"
out_dir = Path(csv_path).parent

print(f"Reading {csv_path}...")
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit(1)

if "cox_p" not in df.columns:
    print("Error: cox_p column not found in CSV")
    print(f"Columns found: {df.columns.tolist()}")
    sys.exit(1)

p_vals = df["cox_p"].dropna()
print(f"Found {len(p_vals)} valid Cox P-values")

# 设置绘图风格
plt.figure(figsize=(8, 6))

# 分箱设置 (仿照之前的 Log-Rank 风格)
bins = [0, 0.001, 0.01, 0.05, 1.0]
labels = ["< 0.001", "0.001 - 0.01", "0.01 - 0.05", "> 0.05"]

cats = pd.cut(p_vals, bins=bins, labels=labels, include_lowest=True, right=True)
counts = cats.value_counts().sort_index()

# 绘制柱状图
# 使用 seaborn 调色板或自定义颜色列表
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']  # 自定义四种颜色
x_labels = counts.index.astype(str)
bars = plt.bar(x_labels, counts.values, color=colors, edgecolor="black", alpha=0.7)

# 添加折线图
plt.plot(x_labels, counts.values, color='gray', marker='o', linestyle='-', linewidth=2, markersize=8)

plt.title("Cox PH P-value Distribution")
plt.xlabel("P-value Range")
plt.ylabel("Count")

# 在柱子上显示数字
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom')

plt.tight_layout()
out_path = out_dir / "cox_p_value_distribution.png"
plt.savefig(out_path, dpi=200)
print(f"Plot saved to {out_path}")
