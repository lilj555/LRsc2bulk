#!/usr/bin/env python3
"""
TCGA PRAD 生存分析脚本
修改版：针对输出层含 Sigmoid 的 GNN 模型，自动执行 Logit 逆变换以优化 Cox 分析。
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import re

# 尝试导入生存分析库
try:
    import lifelines  # type: ignore
    from lifelines import CoxPHFitter, KaplanMeierFitter  # type: ignore
    LIFELINES_AVAILABLE = True
except Exception:
    LIFELINES_AVAILABLE = False

import torch
from sklearn.preprocessing import StandardScaler  # type: ignore

# 处理 torch 安全加载全局变量
try:
    from torch.serialization import add_safe_globals  # type: ignore
except Exception:
    add_safe_globals = None

# 导入你的 GNN 模型构建函数
from src.models.gnn_model import create_model


def setup_logging(output_dir: Path, level: str = "INFO") -> Path:
    """
    设置日志记录到控制台与文件
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "analysis.log"
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    logging.getLogger(__name__).info(f"日志输出: {log_path}")
    return log_path


def normalize_path(p: str) -> Path:
    """规范化路径字符串"""
    return Path(str(p).replace("\\", "/")).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    default_checkpoint = "/home/lilj/work/xenium/code/models/checkpoint_epoch_39.pth"
    parser = argparse.ArgumentParser(
        description="TCGA PRAD lncRNA表达与配体-受体对的生存分析（CoxPH）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint, help="预训练模型检查点")
    parser.add_argument("--clinical", type=str, default="/home/lilj/work/xenium/tcga_data/TCGA-PRAD_clinical.csv", help="临床数据CSV路径")
    parser.add_argument("--expression", type=str, default="/home/lilj/work/xenium/tcga_data/TCGA-PRAD_mrna_expr_tpm.csv", help="mRNA表达TPM数据CSV路径")
    parser.add_argument("--output-dir", type=str, default="/home/lilj/work/xenium/tcga_output", help="结果输出目录")
    parser.add_argument("--lr-file", type=str, default=None, help="自定义LR列表CSV路径")
    parser.add_argument("--preview", action="store_true", help="显示数据预览")
    parser.add_argument("--clinical-id-column", type=str, default=None, help="临床数据样本ID列名")
    parser.add_argument("--expression-id-column", type=str, default=None, help="表达数据样本ID列名")
    parser.add_argument("--id-normalizer", type=str, choices=["tcga_patient", "none"], default="tcga_patient", help="样本ID规范化方式")
    parser.add_argument("--lr-aggregation", type=str, choices=["geomean", "mean", "product"], default="geomean", help="LR表达聚合方式（非GNN模式下）")
    
    # GNN 相关参数
    parser.add_argument("--use-gnn", action="store_true", help="使用训练好的GNN模型对LR进行预测")
    parser.add_argument("--gnn-batch-size", type=int, default=32, help="GNN推理的批大小")
    
    # 统计相关参数
    parser.add_argument("--significance", type=float, default=0.05, help="显著性阈值（CoxPH的p值）")
    parser.add_argument("--cox-penalizer", type=float, default=0.1, help="CoxPHFitter的L2正则强度")
    parser.add_argument("--no-robust", action="store_true", help="关闭CoxPH的robust协方差估计")
    parser.add_argument("--min-strata-var", type=float, default=1e-6, help="事件/非事件分层方差阈值")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="日志级别")
    
    return parser.parse_args()


def load_checkpoint_metadata(checkpoint_path: Path) -> Dict:
    """加载检查点并返回元数据"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到检查点文件: {checkpoint_path}")
    
    if add_safe_globals is not None:
        try:
            add_safe_globals([StandardScaler])
        except Exception:
            pass
            
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if "metadata" not in ckpt:
        raise KeyError("检查点不包含metadata键")
    return ckpt["metadata"]


def parse_lr_pair_name(name: str) -> Optional[Tuple[str, str]]:
    """从LR名称中解析出配体和受体基因符号"""
    if not isinstance(name, str) or len(name) == 0:
        return None
    if "_" in name and "|" in name:
        left, right = name.split("_", 1)
        lig = left.split("|")[-1].strip()
        rec = right.split("|")[-1].strip()
        if lig and rec:
            return lig, rec
    if "|" in name:
        parts = [p.strip() for p in name.split("|") if p.strip()]
        if len(parts) >= 2:
            return parts[-2], parts[-1]
    for sep in ["->", "-", "::", "—", "→", ":", "/", "_"]:
        if sep in name:
            parts = [p.strip() for p in name.split(sep) if p.strip()]
            if len(parts) == 2:
                return parts[0], parts[1]
    return None


def extract_lr_pairs(metadata: Dict) -> List[Tuple[str, str, str]]:
    """提取LR对名称及其基因符号"""
    lr_names = metadata.get("lr_pairs", [])
    if not isinstance(lr_names, list) or len(lr_names) == 0:
        logging.getLogger(__name__).warning("未在metadata中找到lr_pairs列表")
        return []
    pairs = []
    for n in lr_names:
        parsed = parse_lr_pair_name(str(n))
        if parsed is not None:
            pairs.append((str(n), parsed[0], parsed[1]))
    return pairs


def load_lr_file(lr_path: Path) -> List[Tuple[str, str, str]]:
    """从自定义CSV加载LR列表"""
    df = pd.read_csv(lr_path)
    cols = {c.lower(): c for c in df.columns}
    if "ligand" not in cols or "receptor" not in cols:
        raise ValueError("自定义LR文件需包含列: ligand, receptor")
    name_col = cols.get("name", None)
    out = []
    for _, row in df.iterrows():
        lig = str(row[cols["ligand"]]).strip()
        rec = str(row[cols["receptor"]]).strip()
        nm = str(row[name_col]).strip() if name_col else f"{lig}|{rec}"
        if lig and rec:
            out.append((nm, lig, rec))
    return out


def load_clinical(clinical_path: Path, id_column: Optional[str], normalizer: str) -> pd.DataFrame:
    """加载并规范化临床数据"""
    df = pd.read_csv(clinical_path)
    # 识别索引列
    idx_candidates = ["sample_id", "PatientID", "bcr_patient_barcode", "case_submitter_id", "id", "ID"]
    index_col = None
    if id_column and id_column in df.columns:
        index_col = id_column
    else:
        for c in idx_candidates:
            if c in df.columns:
                index_col = c
                break
    
    if index_col is None:
        df.index = df.iloc[:, 0].astype(str)
    else:
        df.index = df[index_col].astype(str)
        
    if normalizer == "tcga_patient":
        def _core_barcode(x: str) -> str:
            s = str(x)
            m = re.search(r"(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})", s, flags=re.IGNORECASE)
            return (m.group(1) if m else s).upper()
        df.index = df.index.map(_core_barcode)
    else:
        df.index = df.index.astype(str).str.upper()
        
    df = df[~df.index.duplicated(keep="first")]
    
    # 标准化生存时间与事件
    cols = {c.lower(): c for c in df.columns}
    duration, event = None, None
    
    d_death = cols.get("days_to_death")
    d_lfu = cols.get("days_to_last_follow_up") or cols.get("days_to_last_followup")
    vital = cols.get("vital_status")
    
    if d_death or d_lfu:
        dd = pd.to_numeric(df[d_death], errors="coerce") if d_death else None
        dl = pd.to_numeric(df[d_lfu], errors="coerce") if d_lfu else None
        if dd is not None and dl is not None:
            duration = dd.fillna(dl)
        elif dd is not None:
            duration = dd
        else:
            duration = dl
            
        if vital:
            vs = df[vital].astype(str).str.lower()
            event = (vs == "dead").astype(int)
        else:
            event = (pd.notna(dd)).astype(int) if dd is not None else pd.Series(0, index=df.index)
            
    if duration is None:
        for cname in ["os.time", "time", "days", "duration"]:
            if cname in cols:
                duration = pd.to_numeric(df[cols[cname]], errors="coerce")
                break
    if event is None:
        for cname in ["os.event", "event", "status"]:
            if cname in cols:
                val = df[cols[cname]]
                if val.dtype == object:
                    vv = val.astype(str).str.lower()
                    event = vv.isin(["1", "dead", "deceased", "true"]).astype(int)
                else:
                    event = pd.to_numeric(val, errors="coerce").fillna(0).astype(int)
                break
                
    if duration is None or event is None:
        raise ValueError("无法识别临床数据中的生存时间/事件字段")
        
    out = pd.DataFrame({"duration": duration, "event": event}, index=df.index)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out["duration"] = out["duration"].clip(lower=0)
    return out


def load_expression(expr_path: Path, id_column: Optional[str], normalizer: str) -> pd.DataFrame:
    """加载表达矩阵"""
    df = pd.read_csv(expr_path)
    first_col = df.columns[0].lower()
    tcga_cols = sum(1 for c in df.columns if re.search(r"TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}", str(c), flags=re.IGNORECASE))
    
    if first_col in ["gene", "gene_symbol", "symbol", "ensembl", "gene_id"] or tcga_cols >= max(5, int(0.3 * len(df.columns))):
        df = df.set_index(df.columns[0])
        df = df.T
    else:
        use_col = id_column if id_column and id_column in df.columns else df.columns[0]
        df.index = df[use_col].astype(str)
        if use_col in df.columns:
            df = df.drop(columns=[use_col])
            
    df.columns = [str(c) for c in df.columns]
    
    if normalizer == "tcga_patient":
        def _core_barcode(x: str) -> str:
            s = str(x)
            m = re.search(r"(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})", s, flags=re.IGNORECASE)
            return (m.group(1) if m else s).upper()
        df.index = df.index.map(_core_barcode)
    else:
        df.index = df.index.astype(str).str.upper()
        
    df = df[~df.index.duplicated(keep="first")]
    df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how="any")
    return df


def compute_lr_expression(expr_df: pd.DataFrame, lr_pairs: List[Tuple[str, str, str]], method: str) -> pd.DataFrame:
    """计算传统统计学方式的LR表达值"""
    out = {}
    genes_lower = {g.lower(): g for g in expr_df.columns}
    for lr_name, lig, rec in lr_pairs:
        lig_key = genes_lower.get(str(lig).lower(), None)
        rec_key = genes_lower.get(str(rec).lower(), None)
        if lig_key is None or rec_key is None:
            continue
        a = expr_df[lig_key]
        b = expr_df[rec_key]
        if method == "product":
            val = a * b
        elif method == "mean":
            val = (a + b) / 2.0
        else:
            val = np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None))
        out[lr_name] = val
    if not out:
        raise ValueError("未能计算任何LR表达")
    return pd.DataFrame(out, index=expr_df.index)


def compute_lr_predictions_with_gnn(expr_df: pd.DataFrame, checkpoint_path: Path, batch_size: int = 32) -> Tuple[pd.DataFrame, dict]:
    """使用GNN模型进行推理"""
    if add_safe_globals is not None:
        try:
            add_safe_globals([StandardScaler])
        except Exception:
            pass
            
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    metadata = ckpt.get("metadata", {})
    state_dict = ckpt.get("model_state_dict", None)
    
    if state_dict is None:
        raise KeyError("检查点缺少 model_state_dict")
        
    model = create_model(config, metadata)
    model.load_state_dict(state_dict)
    model.eval()
    
    gene_to_idx = metadata.get("gene_to_idx", {})
    num_genes = int(metadata.get("num_genes", len(gene_to_idx)))
    scaler = metadata.get("scaler", None)
    
    # 填充特征矩阵
    train_lower = {k.lower(): v for k, v in gene_to_idx.items()}
    expr_cols_lower = {c.lower(): c for c in expr_df.columns}
    idx_and_cols = []
    for gl, idx in train_lower.items():
        colname = expr_cols_lower.get(gl, None)
        if colname is not None:
            idx_and_cols.append((idx, colname))
            
    X = np.zeros((expr_df.shape[0], num_genes), dtype=np.float32)
    for idx, col in idx_and_cols:
        X[:, idx] = expr_df[col].values.astype(np.float32)
        
    if scaler is not None:
        X = scaler.transform(X).astype(np.float32)
        
    edge_index = metadata.get("edge_index", None)
    if edge_index is None:
        raise KeyError("metadata中缺少edge_index")
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
    pathway_mapping = metadata.get("pathway_to_genes", {})
    preds_all = []
    
    with torch.no_grad():
        for start in range(0, X.shape[0], max(1, int(batch_size))):
            end = min(X.shape[0], start + max(1, int(batch_size)))
            x_chunk = torch.from_numpy(X[start:end])
            batch = {
                "gene_expr": x_chunk,
                "edge_index": edge_index,
                "pathway_mapping": pathway_mapping,
            }
            preds = model(batch)
            preds_all.append(preds.detach().cpu().numpy())
            
    pred_np = np.concatenate(preds_all, axis=0)
    lr_names = metadata.get("lr_pairs", [])
    pred_df = pd.DataFrame(pred_np, index=expr_df.index, columns=lr_names)
    
    return pred_df, {"config": config, "metadata": metadata}


def find_best_cutoff(df: pd.DataFrame, col: str, duration_col: str, event_col: str, logrank_func) -> Tuple[float, float]:
    """
    寻找最佳分割点
    返回: (best_cutoff, best_p_value)
    """
    values = df[col].values
    # 使用 20% - 80% 的分位数作为候选点，步长 5%
    candidates = np.unique(np.percentile(values, np.arange(20, 81, 5)))
    
    best_p = 1.0
    best_cut = np.median(values)
    
    for cut in candidates:
        mask = values >= cut
        # 确保每组至少有一定样本量 (比如 10 个或 10%)
        if mask.sum() < 10 or (~mask).sum() < 10:
            continue
            
        try:
            res = logrank_func(
                df[duration_col][mask], df[duration_col][~mask],
                event_observed_A=df[event_col][mask], event_observed_B=df[event_col][~mask]
            )
            if res.p_value < best_p:
                best_p = res.p_value
                best_cut = cut
        except Exception:
            continue
            
    return float(best_cut), float(best_p)


def run_univariate_cox(clin_df: pd.DataFrame, lr_expr_df: pd.DataFrame, penalizer: float, robust: bool, min_strata_var: float) -> pd.DataFrame:
    """执行单变量CoxPH分析，并计算Log-Rank p值（使用最佳分割点）"""
    if not LIFELINES_AVAILABLE:
        raise ImportError("未检测到lifelines，请安装: pip install lifelines")
        
    # 尝试导入 logrank_test
    try:
        from lifelines.statistics import logrank_test
    except ImportError:
        logrank_test = None

    results = []
    base = clin_df.copy()
    events = base["event"].astype(bool)
    
    # 预筛选方差过低的LR
    vars_to_use = []
    for lr in lr_expr_df.columns:
        s = lr_expr_df[lr].reindex(base.index)
        v1 = float(np.nanvar(s[events])) if np.any(events) else 0.0
        v0 = float(np.nanvar(s[~events])) if np.any(~events) else 0.0
        if v1 >= min_strata_var and v0 >= min_strata_var:
            vars_to_use.append(lr)
            
    for lr in vars_to_use:
        df = base.copy()
        df[lr] = lr_expr_df[lr].reindex(df.index)
        df = df.dropna()
        if df[lr].nunique() < 2:
            continue
            
        res_dict = {
            "lr": lr,
            "coef": np.nan,
            "hr": np.nan,
            "p": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "best_cutoff": np.nan,
            "logrank_p": np.nan  # 这里改为存储最佳分割点的p值
        }

        # Cox PH
        try:
            cph = CoxPHFitter(penalizer=float(penalizer))
            cph.fit(df[[lr, "duration", "event"]], duration_col="duration", event_col="event", robust=bool(robust))
            summary = cph.summary.loc[lr]
            
            res_dict.update({
                "coef": float(summary["coef"]),
                "hr": float(summary["exp(coef)"]),
                "p": float(summary["p"]),
                "ci_lower": float(summary["exp(coef) lower 95%"]),
                "ci_upper": float(summary["exp(coef) upper 95%"]),
            })
        except Exception as e:
            logging.getLogger(__name__).debug(f"Cox拟合失败 {lr}: {e}")

        # Log-Rank Test (Best Cutoff)
        if logrank_test is not None:
            try:
                best_cut, best_p = find_best_cutoff(df, lr, "duration", "event", logrank_test)
                res_dict["best_cutoff"] = best_cut
                res_dict["logrank_p"] = best_p
            except Exception as e:
                logging.getLogger(__name__).debug(f"Log-Rank计算失败 {lr}: {e}")
        
        # 只要有一个成功就算数
        if not np.isnan(res_dict["p"]) or not np.isnan(res_dict["logrank_p"]):
             results.append(res_dict)
            
    if not results:
        raise RuntimeError("CoxPH/Log-Rank分析没有可用结果")
    
    # 优先按Cox p排序，如果没有Cox p则按logrank p排序
    res_df = pd.DataFrame(results)
    if "p" in res_df.columns:
        return res_df.sort_values("p")
    return res_df.sort_values("logrank_p")


def save_results_table(results_df: pd.DataFrame, output_dir: Path) -> Path:
    """保存结果CSV"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = output_dir / f"LR_survival_analysis_{ts}.csv"
    results_df.to_csv(save_path, index=False)
    logging.getLogger(__name__).info(f"结果表已保存: {save_path}")
    return save_path


def plot_km_for_lr(clin_df: pd.DataFrame, lr_series: pd.Series, lr_name: str, out_dir: Path, cox_p: Optional[float] = None, cox_hr: Optional[float] = None, cutoff: Optional[float] = None, show_cox_p: bool = True) -> None:
    """绘制KM生存曲线，支持自定义分割点"""
    if not LIFELINES_AVAILABLE:
        return

    # 尝试导入 logrank_test
    try:
        from lifelines.statistics import logrank_test
    except ImportError:
        logrank_test = None
        
    df = clin_df.copy()
    df["lr"] = lr_series.reindex(df.index)
    df = df.dropna()
    if df["lr"].nunique() < 2:
        return
        
    # 确定分割点
    if cutoff is not None:
        thresh = float(cutoff)
        cutoff_source = "Best"
    else:
        # 默认中位数，或者可以在这里重新调用 find_best_cutoff
        # 但为了效率，建议外部传入。如果未传入，这里退化为中位数
        thresh = float(df["lr"].median())
        cutoff_source = "Median"
        
    df["group"] = np.where(df["lr"] >= thresh, "High", "Low")
    
    # 检查分组样本量
    if df["group"].value_counts().min() < 2:
        return

    km = KaplanMeierFitter()
    import matplotlib.pyplot as plt
    
    # 修改为正方形 (6, 6)
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 计算 Log-Rank p值 (基于当前分组)
    logrank_p_text = ""
    if logrank_test is not None:
        try:
            high = df[df["group"] == "High"]
            low = df[df["group"] == "Low"]
            if len(high) > 0 and len(low) > 0:
                res = logrank_test(
                    high["duration"], low["duration"],
                    event_observed_A=high["event"], event_observed_B=low["event"]
                )
                logrank_p_text = f"Log-Rank p = {res.p_value:.2e}"
        except Exception:
            pass

    for grp in ["High", "Low"]:
        sub = df[df["group"] == grp]
        if len(sub) == 0:
            continue
        km.fit(durations=sub["duration"], event_observed=sub["event"], label=f"{grp} (n={len(sub)})")
        km.plot(ax=ax)
        
    # 标题包含 p 值
    title_str = str(lr_name)
    stats_text = []
    
    if show_cox_p and cox_p is not None:
        cox_str = f"Cox p = {cox_p:.2e}"
        if cox_hr is not None:
            cox_str += f", HR = {cox_hr:.2f}"
        stats_text.append(cox_str)
        
    if logrank_p_text:
        stats_text.append(logrank_p_text)
        
    # stats_text.append(f"Cutoff ({cutoff_source}): {thresh:.2f}")
        
    if stats_text:
        title_str += "\n" + "\n".join(stats_text)
        
    ax.set_title(title_str)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.grid(True, alpha=0.3)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    
    clean_name = str(lr_name).replace('/', '_').replace(' ', '_').replace('|', '_')
    png = out_dir / f"{clean_name}.png"
    pdf = out_dir / f"{clean_name}.pdf"
    
    plt.tight_layout()
    plt.savefig(png, dpi=200, bbox_inches="tight")
    plt.savefig(pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.getLogger(__name__).info(f"KM图已保存: {png}")


def plot_p_value_distribution(results_df: pd.DataFrame, out_dir: Path) -> None:
    """绘制P值分布图（柱状图，带数字）"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 关注 Log-Rank P
    if "logrank_p" not in results_df.columns:
        return
        
    p_vals = results_df["logrank_p"].dropna()
    if len(p_vals) == 0:
        return

    # 分箱
    bins = [0, 0.001, 0.01, 0.05, 1.0]
    labels = ["< 0.001", "0.001 - 0.01", "0.01 - 0.05", "> 0.05"]
    
    cats = pd.cut(p_vals, bins=bins, labels=labels, include_lowest=True, right=True)
    counts = cats.value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(counts.index.astype(str), counts.values, color="skyblue", edgecolor="black")
    
    ax.set_title("Log-Rank P-value Distribution")
    ax.set_xlabel("P-value Range")
    ax.set_ylabel("Count")
    
    # 在柱子上显示数字
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
                
    plt.tight_layout()
    plt.savefig(out_dir / "p_value_distribution.png", dpi=200)
    plt.close(fig)


def plot_forest(results_df: pd.DataFrame, out_dir: Path, top_n: int = 20) -> None:
    """绘制Top N的森林图"""
    import matplotlib.pyplot as plt
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = results_df.head(top_n).sort_values("p", ascending=False) # 逆序使得最小p值在最上方
    
    if len(df) == 0:
        return

    y = range(len(df))
    
    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.4)))
    
    ax.errorbar(df["hr"], y, xerr=[df["hr"] - df["ci_lower"], df["ci_upper"] - df["hr"]], 
                fmt='o', color='black', ecolor='gray', capsize=3)
    
    ax.axvline(x=1, linestyle='--', color='red', linewidth=1)
    
    ax.set_yticks(y)
    ax.set_yticklabels(df["lr"])
    ax.set_xlabel("Hazard Ratio (95% CI)")
    ax.set_title(f"Top {len(df)} Significant LR Pairs (Cox PH)")
    
    plt.tight_layout()
    plt.savefig(out_dir / "forest_plot.png", dpi=200)
    plt.close(fig)


def plot_top20_bar(results_df: pd.DataFrame, out_dir: Path, top_n: int = 20) -> None:
    """绘制Top N的 -log10(p) 柱状图 (Log-Rank)"""
    import matplotlib.pyplot as plt
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 确保使用 Log-Rank P
    if "logrank_p" not in results_df.columns:
        return
        
    df = results_df.head(top_n).copy()
    col = "logrank_p"
    
    # 避免p=0导致log无穷大
    min_p = df[df[col] > 0][col].min()
    if pd.isna(min_p): min_p = 1e-300
    df[col] = df[col].replace(0, min_p / 10)
    
    df["log_p"] = -np.log10(df[col])
    df = df.sort_values("log_p", ascending=True) # 使得最显著的在最上面
    
    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.4)))
    
    bars = ax.barh(df["lr"], df["log_p"], color='steelblue')
    
    ax.set_xlabel("-log10(Log-Rank P-value)")
    ax.set_title(f"Top {len(df)} LR Pairs by Log-Rank Significance")
    
    plt.tight_layout()
    plt.savefig(out_dir / "top20_significance_bar.png", dpi=200)
    plt.close(fig)


def main():
    """主入口"""
    args = parse_args()
    output_dir = normalize_path(args.output_dir)
    setup_logging(output_dir, level=args.log_level)
    logger = logging.getLogger(__name__)

    current_env = os.environ.get("CONDA_DEFAULT_ENV", "base")
    if current_env != "deeplearning":
        logger.warning(f"当前conda环境: {current_env}, 期望环境: deeplearning")

    try:
        checkpoint_path = normalize_path(args.checkpoint)
        clinical_path = normalize_path(args.clinical)
        expression_path = normalize_path(args.expression)
        
        if not clinical_path.exists():
            raise FileNotFoundError(f"临床数据不存在: {clinical_path}")
        if not expression_path.exists():
            raise FileNotFoundError(f"表达数据不存在: {expression_path}")

        # 获取LR列表
        lr_pairs: List[Tuple[str, str, str]]
        if args.lr_file:
            lr_pairs = load_lr_file(normalize_path(args.lr_file))
            logger.info(f"使用自定义LR列表，共 {len(lr_pairs)} 对")
        else:
            metadata = load_checkpoint_metadata(checkpoint_path)
            lr_pairs = extract_lr_pairs(metadata)
            logger.info(f"从检查点获得LR对，共 {len(lr_pairs)} 对")
            
        if len(lr_pairs) == 0:
            raise ValueError("未能从检查点中解析到任何LR对")

        # 加载数据
        clin_df = load_clinical(clinical_path, args.clinical_id_column, args.id_normalizer)
        expr_df = load_expression(expression_path, args.expression_id_column, args.id_normalizer)

        if args.preview:
            logger.info(f"临床列名: {list(pd.read_csv(clinical_path, nrows=1).columns)}")
            logger.info(f"临床前5行:\n{pd.read_csv(clinical_path, nrows=5).to_string(index=False)}")
            logger.info(f"样本ID示例: {list(clin_df.index[:3])}")

        # 对齐样本
        common_ids = clin_df.index.intersection(expr_df.index)
        if len(common_ids) == 0:
            raise ValueError("临床与表达数据无样本交集")
        clin_df = clin_df.loc[common_ids]
        expr_df = expr_df.loc[common_ids]
        logger.info(f"有效重叠样本数: {len(common_ids)}")

        # 计算LR值
        if args.use_gnn:
            lr_expr_df, ck = compute_lr_predictions_with_gnn(expr_df, checkpoint_path, batch_size=int(args.gnn_batch_size))
            
            # 【重要修改】针对带Sigmoid输出的模型，进行Logit逆变换
            logger.info("正在对模型输出的概率进行 Logit 逆变换以优化 Cox 分析...")
            
            # 1. 裁剪概率值，防止 log(0) 或 log(1) 产生 inf
            epsilon = 1e-6
            lr_expr_df = lr_expr_df.clip(lower=epsilon, upper=1-epsilon)
            
            # 2. Logit 变换: log(p / (1-p))
            lr_expr_df = np.log(lr_expr_df / (1 - lr_expr_df))
            
        else:
            lr_expr_df = compute_lr_expression(expr_df, lr_pairs, method=args.lr_aggregation)

        # 运行生存分析
        results_df = run_univariate_cox(
            clin_df,
            lr_expr_df,
            penalizer=float(args.cox_penalizer),
            robust=(not args.no_robust),
            min_strata_var=float(args.min_strata_var),
        )
        save_results_table(results_df, output_dir)
        
        # 绘制P值分布
        plot_p_value_distribution(results_df, output_dir)

        # 处理Top 20
        top20_dir = output_dir / "top20_significant"
        top20_dir.mkdir(parents=True, exist_ok=True)
        
        # 优先使用 Log-Rank P 值排序
        sort_col = "logrank_p" if "logrank_p" in results_df.columns else "p"
        sorted_results = results_df.sort_values(sort_col)
        top20_df = sorted_results.head(20)
        
        # 保存Log-Rank专用表格
        if "logrank_p" in results_df.columns:
             lr_save_path = top20_dir / "LR_survival_analysis_logrank_sorted.csv"
             sorted_results.to_csv(lr_save_path, index=False)
             logger.info(f"Log-Rank排序结果表已保存: {lr_save_path}")

        if len(top20_df) > 0:
            logger.info(f"正在生成Top {len(top20_df)} 结果图表到 {top20_dir}...")
            # KM Plots for Top 20
            for _, row in top20_df.iterrows():
                lr = str(row["lr"])
                if lr in lr_expr_df.columns:
                    # Top 20 图中不显示 Cox p
                    plot_km_for_lr(clin_df, lr_expr_df[lr], lr, top20_dir, 
                                   cox_p=row.get("p"), cox_hr=row.get("hr"), 
                                   cutoff=row.get("best_cutoff"),
                                   show_cox_p=False)
            
            # Forest Plot (传入已排序的df)
            plot_forest(sorted_results, top20_dir, top_n=20)
            
            # Bar Plot (传入已排序的df)
            plot_top20_bar(sorted_results, top20_dir, top_n=20)

        # 绘制显著LR的KM图 (此处通常保留 Cox P 筛选，或者也改为 Log-Rank?)
        # 用户的需求主要集中在 Top 20 文件夹。这里是 "km_plots" 文件夹。
        # 为了保持一致性，如果用户更看重 Log-Rank，这里也可以改为 Log-Rank 筛选。
        # 既然之前的代码已经支持根据 sort_col 筛选，这里我们继续使用 results_df 原始筛选逻辑，
        # 但因为 sort_col 可能是 logrank_p，我们需要确保 sig 的定义一致。
        
        if "logrank_p" in results_df.columns:
             sig = results_df[results_df["logrank_p"] <= float(args.significance)]
        elif "p" in results_df.columns:
             sig = results_df[results_df["p"] <= float(args.significance)]
        else:
             sig = pd.DataFrame()
             
        if len(sig) > 0:
            km_dir = output_dir / "km_plots"
            logger.info(f"发现 {len(sig)} 个显著LR (p<={args.significance})，开始绘制KM图...")
            for _, row in sig.iterrows():
                lr = str(row["lr"])
                if lr in lr_expr_df.columns:
                    # 普通 KM 图可以保留 Cox P
                    plot_km_for_lr(clin_df, lr_expr_df[lr], lr, km_dir, cox_p=row.get("p"), cox_hr=row.get("hr"), cutoff=row.get("best_cutoff"), show_cox_p=True)
        else:
            logger.info("未检出显著LR（p<=阈值），跳过KM绘制")

        logger.info("分析完成")
        
    except Exception as e:
        logger.error(f"执行出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
