"""Predict with the archived model; never fit or select a model."""
from pathlib import Path
import json
import argparse
import hashlib
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
# Input tables are supplied separately from this code/model directory.
FEATURES = [f'feature{i}' for i in range(2, 11)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args=parser.parse_args()
    INPUT=args.data_dir
    REFERENCE=args.data_dir
    OUTPUT=args.output_dir
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_path = ROOT / 'model.joblib'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = joblib.load(model_path)
    raw = pd.read_csv(INPUT / 'candidates.csv')
    X = raw[FEATURES]
    if not np.isfinite(X.to_numpy(float)).all():
        raise ValueError('Candidate features include non-finite values')
    results = raw.iloc[:, :2].copy()
    results.columns = ['record_id', 'record_name']
    results['predicted_barrier_eV'] = model.predict(X)
    results.to_csv(OUTPUT / 'predictions.csv', index=False, encoding='utf-8-sig')
    results['structure_group'] = results['record_name'].str.split('_pro').str[0]
    filtered = results.loc[results.groupby('structure_group')['predicted_barrier_eV'].idxmin()].copy()
    ranked = filtered.sort_values('predicted_barrier_eV', ascending=False).copy()
    ranked['rank'] = np.arange(1, len(ranked) + 1)
    ranked = ranked[['rank', 'record_id', 'record_name', 'predicted_barrier_eV']]
    ranked.to_csv(OUTPUT / 'ranked_predictions.csv', index=False, encoding='utf-8-sig')
    reference = pd.read_csv(REFERENCE / 'reference_predictions.csv')
    old_rank = pd.read_csv(REFERENCE / 'reference_ranking.csv')
    key = ['record_id', 'record_name']
    merged = results.merge(reference[key + ['predicted_barrier_eV']], on=key,
                           suffixes=('_new', '_saved'), validate='one_to_one', how='outer', indicator=True)
    matched = bool((merged['_merge'] == 'both').all())
    difference = np.abs(merged['predicted_barrier_eV_new'] - merged['predicted_barrier_eV_saved'])
    max_error = float(difference.max()) if matched else None
    same_order = (ranked['record_name'].astype(str).tolist() == old_rank['record_name'].astype(str).tolist())
    tied_rank_equivalent = len(ranked) == len(old_rank) and bool(np.allclose(
        ranked['predicted_barrier_eV'].to_numpy(), old_rank['predicted_barrier_eV'].to_numpy(), rtol=0, atol=1e-10))
    rank_mismatches = int(np.count_nonzero(ranked['record_name'].to_numpy() != old_rank['record_name'].to_numpy()))
    saved_records = reference.copy()
    saved_records['structure_group'] = saved_records['record_name'].str.split('_pro').str[0]
    saved_rank = saved_records.loc[saved_records.groupby('structure_group')['predicted_barrier_eV'].idxmin()].sort_values(
        'predicted_barrier_eV', ascending=False)
    saved_scores_reproduce_order = saved_rank['record_name'].tolist() == old_rank['record_name'].tolist()
    metrics = {}
    for label, filename in [('training', 'training.parquet'), ('historical_holdout', 'holdout.parquet')]:
        frame = pd.read_parquet(INPUT / filename)
        prediction = model.predict(frame[FEATURES])
        metrics[label] = dict(rows=len(frame), mae=float(mean_absolute_error(frame.target, prediction)),
                              rmse=float(np.sqrt(mean_squared_error(frame.target, prediction))),
                              r2=float(r2_score(frame.target, prediction)))
    report = dict(candidate_rows=len(raw), structure_groups=len(ranked),
                  candidate_keys_match=matched, max_abs_prediction_difference_eV=max_error,
                  predictions_match_within_1e_10_eV=bool(matched and max_error <= 1e-10),
                  entire_ranking_order_matches=same_order,
                  ranking_equivalent_within_1e_10_eV=tied_rank_equivalent,
                  rank_positions_different_due_to_numerical_ties=rank_mismatches,
                  top8_names_and_order_match=ranked.head(8)['record_name'].tolist() == old_rank.head(8)['record_name'].tolist(),
                  archived_record_scores_reproduce_exact_ranking=saved_scores_reproduce_order,
                  loaded_model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  original_ssl_model_sha256=hashlib.sha256((ROOT / 'original_ssl_model.joblib').read_bytes()).hexdigest(),
                  metrics=metrics, model_load_warnings=sorted(set(str(w.message) for w in caught)),
                  training_performed=False)
    (OUTPUT / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['predictions_match_within_1e_10_eV'] or not tied_rank_equivalent or not saved_scores_reproduce_order:
        raise SystemExit('Historical prediction/ranking mismatch; see verification.json')


if __name__ == '__main__':
    main()
