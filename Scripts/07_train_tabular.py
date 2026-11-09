#!/usr/bin/env python3
"""AMLworld HI-Small tabular benchmarks: SGD logistic regression, XGBoost, XGBoost+train-graph statistics.
Input: ~/Bank/graph/hi_small/tensors/{train,validation,test}.pt
Never tunes on test labels. Run from ~/Bank or any directory.
"""
import argparse
import csv
import json
import time
import joblib
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

SPLITS = ('train', 'validation', 'test')


def load_splits(root):
    tensors = root / 'graph' / 'hi_small' / 'tensors'
    split = {name: torch.load(tensors / f'{name}.pt', map_location='cpu', weights_only=True) for name in SPLITS}
    expected = {'train': (3248921,2530), 'validation': (965524,1036), 'test': (863900,1611)}
    for name, d in split.items():
        n, positives = expected[name]
        assert len(d['y']) == n and int(d['y'].sum()) == positives, f'Unexpected {name} dataset'
        assert d['edge_index'].shape == (2,n)
    return split


def node_stats(train, n_nodes=515088):
    """Only training graph (features, no labels): source/destination counts and amount totals."""
    src = train['edge_index'][0].numpy()
    dst = train['edge_index'][1].numpy()
    amt = np.maximum(train['edge_amounts'].numpy().astype(np.float64), 0)
    stats = np.column_stack([
        np.log1p(np.bincount(src, minlength=n_nodes)),
        np.log1p(np.bincount(dst, minlength=n_nodes)),
        np.log1p(np.bincount(src, weights=amt[:,0], minlength=n_nodes)),
        np.log1p(np.bincount(dst, weights=amt[:,1], minlength=n_nodes))
    ]).astype(np.float32)
    return stats


def tabular_features(d, stats=None):
    amt = np.maximum(d['edge_amounts'].numpy(), 0)
    cat = d['edge_categories'].numpy()
    src, dst = d['edge_index'].numpy()
    secs = d['edge_timestamps'].numpy().astype(np.float32)
    hour = (secs / 3600.0) % 24
    xnum = np.column_stack([
        np.log1p(amt[:, 0]), np.log1p(amt[:, 1]),
        np.abs(np.log1p(amt[:, 0]) - np.log1p(amt[:, 1])),
        (cat[:,0] == cat[:,1]).astype(np.float32),
        (src == dst).astype(np.float32),
        np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
    ]).astype(np.float32)
    # Train-derived vocab; 0 reserved for unseen categories.
    cats = np.concatenate([np.eye(k,dtype=np.float32)[cat[:,j]] for j,k in enumerate((16,16,8))],axis=1)
    X = np.concatenate([xnum,cats],axis=1)
    if stats is not None:
        X = np.concatenate([X, stats[src], stats[dst]],axis=1)
    return X


def choose_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    i = int(np.argmax(f))
    return float(thresholds[i]), float(f[i])


def metrics(y, p, thr):
    pred = (p >= thr).astype(np.int8)
    cm = confusion_matrix(y,pred,labels=[0,1])
    return {
        'ap_pr_auc': float(average_precision_score(y,p)),
        'roc_auc': float(roc_auc_score(y,p)),
        'precision':float(precision_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0)),
        'f1':float(f1_score(y,pred,zero_division=0)),
        'tn':int(cm[0,0]),'fp':int(cm[0,1]),'fn':int(cm[1,0]),'tp':int(cm[1,1]),
        'threshold':float(thr)
    }


def record(name,pval,ptest,yval,ytest,out, elapsed, test_timestamps=None):
    threshold, val_best = choose_threshold(yval,pval)
    val = metrics(yval,pval,threshold)
    test = metrics(ytest,ptest,threshold)
    obj={'model':name,'threshold_selected_from':'validation_F1','validation':val,'test':test,'elapsed_sec':round(elapsed,1)}
    if test_timestamps is not None:
        late=np.asarray(test_timestamps)>=10*86400  # Sept 11 onward, seconds since Sept 1
        obj['test_by_period']={}
        for period,mask in [('sep09_10',~late),('sep11_18',late)]:
            if mask.any():
                obj['test_by_period'][period]={'n':int(mask.sum()),'positives':int(ytest[mask].sum()),**metrics(ytest[mask],ptest[mask],threshold)}
    out.mkdir(parents=True,exist_ok=True)
    with open(out/f'{name}.json','w') as f: json.dump(obj,f,indent=2)
    print(f'\n{name} | VALIDATION AP={val["ap_pr_auc"]:.5f} F1={val["f1"]:.5f} | TEST AP={test["ap_pr_auc"]:.5f} F1={test["f1"]:.5f} P={test["precision"]:.5f} R={test["recall"]:.5f} | threshold={threshold:.6f}',flush=True)
    return obj


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path.home()/'Bank')
    ap.add_argument('--models', default='lr,xgb,xgb_graph',help='comma-separated: lr,xgb,xgb_graph')
    ap.add_argument('--xgb-device',default='cuda',choices=['cuda','cpu'])
    ap.add_argument('--xgb-trees',type=int,default=220)
    args=ap.parse_args()
    out=args.root/'results'/'hi_small'
    models=[s.strip() for s in args.models.split(',')]
    print('Loading complete IBM tensors...',flush=True)
    splits=load_splits(args.root)
    y={k:d['y'].numpy().astype(np.int32) for k,d in splits.items()}
    graphstats=node_stats(splits['train']) if 'xgb_graph' in models else None
    results=[]

    if 'lr' in models:
        t=time.time()
        X={k:tabular_features(d) for k,d in splits.items()}
        sc=StandardScaler()
        Xtrain=sc.fit_transform(X['train'])
        model=SGDClassifier(loss='log_loss',class_weight='balanced',alpha=1e-5,
                            max_iter=30,tol=1e-3,random_state=42,average=True)
        print('Training Logistic Regression (SGD solver)...',flush=True)
        model.fit(Xtrain,y['train'])
        pv=model.predict_proba(sc.transform(X['validation']))[:,1]
        pt=model.predict_proba(sc.transform(X['test']))[:,1]
        results.append(record('logistic_regression',pv,pt,y['validation'],y['test'],out,time.time()-t, splits['test']['edge_timestamps'].numpy()))
        joblib.dump({'scaler':sc,'model':model},out/'logistic_regression.joblib')
        del X,Xtrain,pv,pt

    if any(k in models for k in ('xgb','xgb_graph')):
        import xgboost as xgb
        for name,use_graph in [('xgb',False),('xgb_graph',True)]:
            if name not in models:continue
            t=time.time()
            print(f'Training {name} on {args.xgb_device}...',flush=True)
            X={k:tabular_features(d,graphstats if use_graph else None) for k,d in splits.items()}
            npos=int(y['train'].sum()); nneg=len(y['train'])-npos
            params=dict(n_estimators=args.xgb_trees,max_depth=6,learning_rate=0.075,
                        subsample=0.8,colsample_bytree=0.9,min_child_weight=5,
                        objective='binary:logistic',eval_metric='aucpr',
                        scale_pos_weight=nneg/npos,tree_method='hist',
                        device=args.xgb_device,random_state=42,n_jobs=8)
            model=xgb.XGBClassifier(**params)
            try:
                model.fit(X['train'],y['train'],verbose=False)
            except xgb.core.XGBoostError as exc:
                if args.xgb_device != 'cuda':raise
                print(f'CUDA XGBoost failed ({str(exc)[:140]}). Retrying on CPU...',flush=True)
                model=xgb.XGBClassifier(**{**params,'device':'cpu'})
                model.fit(X['train'],y['train'],verbose=False)
            pv=model.predict_proba(X['validation'])[:,1]
            pt=model.predict_proba(X['test'])[:,1]
            results.append(record(name,pv,pt,y['validation'],y['test'],out,time.time()-t, splits['test']['edge_timestamps'].numpy()))
            model.save_model(str(out/f'{name}.ubj'))
            del X,pv,pt,model

    if results:
        with open(out/'tabular_summary.csv','w',newline='') as f:
            w=csv.writer(f)
            w.writerow(['model','validation_AP','validation_F1','test_AP','test_ROC_AUC','test_P','test_R','test_F1','threshold','elapsed_sec'])
            for r in results:
                t=r['test'];v=r['validation']
                w.writerow([r['model'],v['ap_pr_auc'],v['f1'],t['ap_pr_auc'],t['roc_auc'],t['precision'],t['recall'],t['f1'],t['threshold'],r['elapsed_sec']])
        print(f'\nSaved results to {out}/tabular_summary.csv',flush=True)

if __name__=='__main__':main()
