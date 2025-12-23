import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import seaborn as sns

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
    sys.exit(1)

# 取 Top 20 显著 (Cox P)
top_n = 20
df_sorted = df.sort_values("cox_p", ascending=True).head(top_n).copy()

# 处理 P 值转换
min_p = df[df["cox_p"] > 0]["cox_p"].min()
if pd.isna(min_p): min_p = 1e-300
df_sorted["cox_p"] = df_sorted["cox_p"].replace(0, min_p / 10)
df_sorted["log_p"] = -np.log10(df_sorted["cox_p"])

# 重新排序以便绘图（最显著的在最上面）
df_plot = df_sorted.sort_values("log_p", ascending=True)

# 绘图
fig, ax = plt.subplots(figsize=(6, 8))

# 生成颜色 (多彩)
colors = sns.color_palette("hls", len(df_plot))

# 绘制棒棒糖图 (Lollipop Chart)
# 1. 绘制线 (stem)
ax.hlines(y=df_plot["lr"], xmin=0, xmax=df_plot["log_p"], color='gray', alpha=0.5, linewidth=1)

# 2. 绘制点 (marker) - 使用多彩颜色
for i, (y, x) in enumerate(zip(df_plot["lr"], df_plot["log_p"])):
    # 显式使用关键字参数，避免将字符串解析为格式化字符串
    ax.plot([x], [y], marker='o', markersize=8, color=colors[i], markeredgecolor='black', linestyle='None')

ax.set_xlabel("-log10(Cox P-value)")
ax.set_title(f"Top {len(df_plot)} LR Pairs by Cox Significance")
ax.grid(axis='x', linestyle='--', alpha=0.3)

# 调整布局
plt.tight_layout()
out_path = out_dir / "top20_significance_lollipop.png"
plt.savefig(out_path, dpi=200)
print(f"Plot saved to {out_path}")
