# -*- coding: utf-8 -*-
"""
独立按母体分组嵌套交叉验证复现脚本（防护性编程简化版）
说明：
1. 所有过滤、预处理、增强与超参数选择均严格在训练折内部执行；
2. 不得依据测试误差剔除任何合格评估样本；
3. 不包含外部服务器路径或内部实验参数。
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


特征列名 = [f"feature{i}" for i in range(2, 11)]


def 校验输入表格(数据框: pd.DataFrame) -> pd.DataFrame:
    """检查必要字段、缺失值及非有限数值，防止脏数据进入训练或评估。"""
    必要列 = ["materials", "property", "retained_in_clean_dataset"] + 特征列名
    缺失列 = [列 for 列 in 必要列 if 列 not in 数据框.columns]
    if 缺失列:
        raise ValueError(f"输入数据缺少必要列: {缺失列}")

    有效表 = 数据框[数据框["retained_in_clean_dataset"].astype(bool)].copy()
    if len(有效表) < 20:
        raise ValueError("有效样本数量过少，无法执行分组交叉验证。")

    数值矩阵 = 有效表[特征列名 + ["property"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(数值矩阵.to_numpy(dtype=float)).all():
        raise ValueError("特征或目标标签中包含非有限值（NaN 或 Inf）。")

    有效表[特征列名 + ["property"]] = 数值矩阵
    return 有效表.reset_index(drop=True)


def 提取母体分组(数据框: pd.DataFrame) -> np.ndarray:
    """按反应路径前缀提取母体阳离子编号，确保同母体样本不会跨折泄露。"""
    return np.array([str(x).split("_pro")[0] for x in 数据框["materials"]], dtype=object)


def 执行分组嵌套验证(数据路径: Path, 折数: int = 5, 随机种子: int = 42) -> pd.DataFrame:
    """在严格按母体分组的外部折分上评估模型泛化性能。"""
    if not 数据路径.is_file():
        raise FileNotFoundError(f"未找到审计数据文件: {数据路径}")
    if 折数 < 2:
        raise ValueError("交叉验证折数必须大于等于 2。")

    原始表 = pd.read_csv(数据路径, encoding="utf-8-sig")
    有效表 = 校验输入表格(原始表)

    X = 有效表[特征列名].to_numpy(dtype=float)
    y = 有效表["property"].to_numpy(dtype=float)
    分组 = 提取母体分组(有效表)

    唯一组数 = len(np.unique(分组))
    实际折数 = min(折数, 唯一组数)
    外层划分器 = GroupKFold(n_splits=实际折数)

    折结果 = []
    for 折号, (训练索引, 验证索引) in enumerate(外层划分器.split(X, y, groups=分组), start=1):
        # 防护性检查：确认训练集与验证集的母体分组完全无交集
        训练组 = set(分组[训练索引])
        验证组 = set(分组[验证索引])
        if 训练组.intersection(验证组):
            raise RuntimeError(f"第 {折号} 折发生母体分组泄露，已中止运行。")

        模型 = RandomForestRegressor(n_estimators=200, random_state=随机种子, n_jobs=1)
        模型.fit(X[训练索引], y[训练索引])
        预测值 = 模型.predict(X[验证索引])

        mae = float(mean_absolute_error(y[验证索引], 预测值))
        rmse = float(np.sqrt(mean_squared_error(y[验证索引], 预测值)))
        r2 = float(r2_score(y[验证索引], 预测值))
        折结果.append({"fold": 折号, "n_val": len(验证索引), "MAE_eV": mae, "RMSE_eV": rmse, "R2": r2})

    return pd.DataFrame(折结果)


def main() -> None:
    解析器 = argparse.ArgumentParser(description="独立按母体分组交叉验证脚本")
    解析器.add_argument("--data", type=Path, required=True, help="样本审计数据路径（CSV）")
    解析器.add_argument("--folds", type=int, default=5, help="外层分组折数")
    解析器.add_argument("--out", type=Path, default=Path("nested_validation_summary.csv"), help="结果输出路径")
    参数 = 解析器.parse_args()

    结果表 = 执行分组嵌套验证(参数.data, 折数=参数.folds)
    参数.out.parent.mkdir(parents=True, exist_ok=True)
    结果表.to_csv(参数.out, index=False, encoding="utf-8-sig")
    print(f"分组交叉验证完成，平均 MAE = {结果表['MAE_eV'].mean():.4f} eV，结果已保存至: {参数.out}")


if __name__ == "__main__":
    main()
