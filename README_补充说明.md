# 第三次返修补充归档说明

本补充包用于对应审稿意见第 6 条（可复现性与模型可追溯性），所有脚本均采用中文注释与防护性编程规范整理，不含内部服务器路径或敏感信息。

## 目录结构

1. `manifest.csv`：全流程关联清单，记录结构编号、各路径对应产物、能垒标签、资格审核与剔除原因、分组交叉验证键、量子化学参考目录及对应验证脚本。
2. `9_Validation_and_Traceability/`：
   - `run_nested_validation.py`：严格按母体分组的独立交叉验证脚本（所有过滤与拟合均在训练折内完成）。
   - `run_ranking_uncertainty.py`：筛选模型（`8_Prediction/model.joblib`，即基于半监督增强训练集拟合的随机森林，无特征标准化）推理与排序稳定性复现脚本。
   - `recompute_barrier_summary.py`：构象及活化自由能能垒复算脚本。
3. `Data_Supplement/`：
   - `sample_eligibility_audit.csv`：样本资格审计与剔除记录。
   - `oof_error_calibration_bins.csv`：折外误差校准分箱统计。
   - `candidate_ranking_stability.csv`：候选结构排名稳定性统计。
   - `barrier_conformer_scan.tsv`：构象与热力学自由能扫描汇总。
   - `barrier_method_reconciliation.csv`：图 7 与图 S22 能垒方法差异分解对照。
