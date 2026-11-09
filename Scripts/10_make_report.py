#!/usr/bin/env python3
"""Collect per-model JSON outputs into a Markdown/CSV comparison report."""
import csv
import json
from pathlib import Path

out=Path.home()/'Bank'/'results'/'hi_small'
expected=['logistic_regression','xgb','xgb_graph','sage','sage_performer','fedavg_sage_performer']
rows=[]
for name in expected:
    p=out/f'{name}.json'
    if not p.is_file():continue
    r=json.loads(p.read_text())
    v=r['validation'];t=r['test']
    rows.append({
        'Model':name,
        'Val AP':v['ap_pr_auc'],
        'Val F1':v['f1'],
        'Test AP':t['ap_pr_auc'],
        'Test ROC-AUC':t['roc_auc'],
        'Test Precision':t['precision'],
        'Test Recall':t['recall'],
        'Test F1':t['f1'],
        'Threshold (val)':t['threshold'],
        'Elapsed seconds':r.get('elapsed_sec',0),
    })
if not rows:raise FileNotFoundError(f'No model results found in {out}')
out.mkdir(parents=True,exist_ok=True)
with open(out/'model_comparison.csv','w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
md=['# IBM AML HI-Small — Model comparison','',
    'Results were produced on local IBM AMLworld data, using chronological IBM-style partitions.','',
    '| Model | Validation AP | Test AP | Test Precision | Test Recall | Test F1 |',
    '|---|---:|---:|---:|---:|---:|']
for r in rows:
    md.append(f'| {r["Model"]} | {r["Val AP"]:.4f} | {r["Test AP"]:.4f} | {r["Test Precision"]:.4f} | {r["Test Recall"]:.4f} | {r["Test F1"]:.4f} |')
md += ['', '## Evaluation protocol', '',
       'Train: Sep 1–6, 2022; validation: Sep 7–8; test: Sep 9–18.',
       'Each decision threshold was tuned exclusively on validation labels to maximize F1.',
       'Graph models use cumulative graph snapshots and *train-only* activity aggregates.',
       'Model selection uses validation average precision; test labels are used for final reporting only.',
       'The Sep 11–18 test tail has atypically high laundering prevalence; consult per-model JSON `test_by_period`.',
       'FedAvg uses simulated bank consortia, not real federated bank infrastructure.',
       'This is a new reimplementation, not the original recovered code.',
       '']
(out/'model_comparison.md').write_text('\n'.join(md))
print('\n'.join(md))
print(f'\nSaved: {out}/model_comparison.csv and {out}/model_comparison.md')
