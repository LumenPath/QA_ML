# -*- coding: utf-8 -*-
"""
活化自由能能垒与构象参数复算脚本（防护性编程简化版）
说明：
1. 从归档的热力学数据表复算吉布斯活化自由能能垒（Ha 与 eV）；
2. 校验反应物、过渡态与氢氧根能量完整性，防止缺失或单位混淆。
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


哈特里转电子伏特 = 27.211386245988


def 复算构象能垒(输入路径: Path) -> pd.DataFrame:
    """读取归档的构象自由能表并核验 ΔG‡ = G(TS) - G(Cat) - G(OH-)。"""
    if not 输入路径.is_file():
        raise FileNotFoundError(f"未找到构象自由能文件: {输入路径}")

    分隔符 = "\t" if 输入路径.suffix.lower() == ".tsv" else ","
    数据框 = pd.read_csv(输入路径, sep=分隔符)

    必要列 = ["system", "site", "conformer", "G_cat_Ha", "G_TS_Ha", "G_OH_Ha"]
    缺失列 = [c for c in 必要列 if c not in 数据框.columns]
    if 缺失列:
        raise ValueError(f"输入表缺少必要列: {缺失列}")

    数值列 = ["G_cat_Ha", "G_TS_Ha", "G_OH_Ha"]
    数值矩阵 = 数据框[数值列].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(数值矩阵.to_numpy(dtype=float)).all():
        raise ValueError("自由能列包含非有限数值。")

    结果 = 数据框.copy()
    结果["recomputed_dG_Ha"] = 数值矩阵["G_TS_Ha"] - 数值矩阵["G_cat_Ha"] - 数值矩阵["G_OH_Ha"]
    结果["recomputed_dG_eV"] = 结果["recomputed_dG_Ha"] * 哈特里转电子伏特
    return 结果


def main() -> None:
    解析器 = argparse.ArgumentParser(description="构象活化自由能复算脚本")
    解析器.add_argument("--input", type=Path, required=True, help="构象自由能汇总表（TSV/CSV）")
    解析器.add_argument("--out", type=Path, default=Path("recomputed_barriers.csv"), help="输出文件路径")
    参数 = 解析器.parse_args()

    输出表 = 复算构象能垒(参数.input)
    参数.out.parent.mkdir(parents=True, exist_ok=True)
    输出表.to_csv(参数.out, index=False, encoding="utf-8-sig")
    print(f"自由能能垒复算完成，共处理 {len(输出表)} 条构象记录，结果已保存至: {参数.out}")


if __name__ == "__main__":
    main()
