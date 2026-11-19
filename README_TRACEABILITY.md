# Traceability and Reproducibility Supplement

This supplement addresses reproducibility and model traceability:

1. manifest.csv: Master traceability manifest linking parent structures, pathway-specific products, barrier labels, eligibility/exclusion status, parent-grouped cross-validation keys, quantum-chemical reference directories, and validation/ranking/barrier scripts.
2. 9_Validation_and_Traceability/: Scripts for parent-grouped validation (run_nested_validation.py), screening inference and ranking stability without feature scaling (run_ranking_uncertainty.py), and activation free-energy barrier recomputation (recompute_barrier_summary.py).
3. Data_Supplement/: Sanitized audit and benchmark tables (sample_eligibility_audit.csv, oof_error_calibration_bins.csv, candidate_ranking_stability.csv, barrier_conformer_scan.tsv, and barrier_method_reconciliation.csv).
