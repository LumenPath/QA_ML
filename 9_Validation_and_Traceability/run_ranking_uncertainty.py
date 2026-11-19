# -*- coding: utf-8 -*-
"""
筛选模型推理与排序稳定性评估脚本（防护性编程简化版）
说明：
1. 直接加载已保存的半监督增强随机森林模型（不做特征标准化）；
2. 严格按母体结构分组取最低预测能垒后降序排序；
3. 支持基于自助重采样评估候选排名的稳定性区间。
"""

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def 加载并校验模型(模型路径: Path):
    """安全加载序列化模型并确认其为树集成预测器。"""
    if not 模型路径.is_file():
        raise FileNotFoundError(f"未找到模型文件: {模型路径}")
    模型 = joblib.load(模型路径)
    if not hasattr(模型, "estimators_") or not hasattr(模型, "feature_names_in_"):
        raise TypeError("加载的模型缺少 estimators_ 或 feature_names_in_ 属性，无法进行树间离散度评估。")
    return 模型


def 校验候选特征矩阵(候选表: pd.DataFrame, 特征名列表: list) -> np.ndarray:
    """按模型记录的原始物理特征顺序提取矩阵，不做标准化，并校验非有限值。"""
    缺失特征 = [c for c in 特征名列表 if c not in 候选表.columns]
    if 缺失特征:
        raise ValueError(f"候选数据缺少模型所需特征列: {缺失特征}")

    矩阵 = 候选表[特征名列表].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(矩阵).all():
        raise ValueError("候选特征矩阵存在缺失或非有限数值。")
    return 矩阵


def 计算预测与排序稳定性(
    模型路径: Path,
    候选表路径: Path,
    名称列索引: int = 1,
    重采样次数: int = 500,
    随机种子: int = 2026,
) -> pd.DataFrame:
    """复现点预测排序并输出自助重采样排名置信区间。"""
    if not 候选表路径.is_file():
        raise FileNotFoundError(f"未找到候选表文件: {候选表路径}")
    if 重采样次数 < 10:
        raise ValueError("重采样次数至少为 10。")

    模型 = 加载并校验模型(模型路径)
    特征列 = list(模型.feature_names_in_)

    候选表 = pd.read_csv(候选表路径, encoding="utf-8", encoding_errors="replace")
    if 名称列索引 >= len(候选表.columns):
        raise IndexError("名称列索引超出候选表列数范围。")

    X = 校验候选特征矩阵(候选表, 特征列)
    单树预测 = np.vstack([树.predict(X) for 树 in 模型.estimators_])
    点预测 = 单树预测.mean(axis=0)
    树间标准差 = 单树预测.std(axis=0, ddof=0)

    名称序列 = 候选表.iloc[:, 名称列索引].astype(str)
    母体分组 = 名称序列.str.split("_pro").str[0]

    结果 = pd.DataFrame({
        "filename": 名称序列,
        "structure_group": 母体分组,
        "prediction_eV": 点预测,
        "tree_sd_eV": 树间标准差,
    })

    # 按母体结构取最薄弱反应路径（最低预测能垒）
    最小索引 = 结果.groupby("structure_group", sort=False)["prediction_eV"].idxmin()
    汇总 = 结果.loc[最小索引].sort_values("prediction_eV", ascending=False, kind="stable").reset_index(drop=True)
    汇总["point_rank"] = np.arange(1, len(汇总) + 1)

    # 基于树间离散度执行正态扰动自助重采样，评估前十保留率与排名区间
    随机数 = np.random.default_rng(随机种子)
    基准值 = 汇总["prediction_eV"].to_numpy(dtype=float)
    标准差 = 汇总["tree_sd_eV"].to_numpy(dtype=float)
    排名记录 = np.empty((重采样次数, len(汇总)), dtype=int)

    for 次 in range(重采样次数):
        扰动值 = 基准值 + 随机数.normal(0.0, 标准差)
        排序下标 = np.argsort(-扰动值, kind="stable")
        名次 = np.empty(len(汇总), dtype=int)
        名次[排序下标] = np.arange(1, len(汇总) + 1)
        排名记录[次] = 名次

    汇总["top10_frequency"] = (排名记录 <= 10).mean(axis=0)
    汇总["bootstrap_rank_p05"] = np.percentile(排名记录, 5, axis=0)
    汇总["bootstrap_rank_p95"] = np.percentile(排名记录, 95, axis=0)
    return 汇总


def main() -> None:
    解析器 = argparse.ArgumentParser(description="筛选模型预测与排名稳定性复现脚本")
    解析器.add_argument("--model", type=Path, required=True, help="已保存模型路径（joblib）")
    解析器.add_argument("--candidates", type=Path, required=True, help="候选结构特征表（CSV）")
    解析器.add_argument("--out", type=Path, default=Path("ranking_stability_reproduced.csv"), help="输出文件路径")
    参数 = 解析器.parse_args()

    输出表 = 计算预测与排序稳定性(参数.model, 参数.candidates)
    参数.out.parent.mkdir(parents=True, exist_ok=True)
    输出表.to_csv(参数.out, index=False, encoding="utf-8-sig")
    print(f"排名稳定性复现完成，共 {len(输出表)} 个母体结构，结果已保存至: {参数.out}")


if __name__ == "__main__":
    main()
