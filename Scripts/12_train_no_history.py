#!/usr/bin/env python3
"""Controlled no-history ablation for 11_train_temporal_edge_gnn.py.

Uses the identical current-transaction encoder, account projection, classifier,
loss, optimizer and train/validation/test protocol as edge_history_mean, but
removes ALL historical transaction inputs and history-count features.
Does not modify the original experiment or overwrite its results.
"""
import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'temporal_edge_gnn', HERE / '11_train_temporal_edge_gnn.py')
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)


class CurrentOnlyEvents(model_module.TemporalEdges):
    """Expose current edge features, replace all endpoint history with zeros."""

    def batch(self, ids, k, device):
        ids = np.asarray(ids, dtype=np.int64)
        b = len(ids)
        current_num = torch.as_tensor(
            np.ascontiguousarray(self.num[ids]), dtype=torch.float32,
            device=device)
        current_cat = torch.as_tensor(
            np.ascontiguousarray(self.cat[ids].astype(np.int64)),
            dtype=torch.long, device=device)
        return (
            current_num,
            current_cat,
            torch.zeros(b, 2, k, 7, device=device),
            torch.zeros(b, 2, k, 3, dtype=torch.long, device=device),
            torch.zeros(b, 2, k, 3, device=device),
            torch.zeros(b, 2, k, dtype=torch.bool, device=device),
            torch.zeros(b, 2, device=device),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.home() / 'Bank')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--eval-batch-size', type=int, default=2048)
    parser.add_argument('--history-k', type=int, default=12,
                        help='Placeholder sequence length; all history is masked out')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--negatives-per-positive', type=int, default=20)
    parser.add_argument('--pos-weight', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--val-proxy-negatives', type=int, default=100000)
    parser.add_argument('--validate-every', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, flush=True)
    print('Loading IBM splits; disabling ALL historical features...', flush=True)
    store = CurrentOnlyEvents(model_module.tab.load_splits(args.root))
    output = args.root / 'results' / 'hi_small'
    output.mkdir(parents=True, exist_ok=True)
    name = 'edge_history_none'
    result = model_module.train_one(name, store, args, device, output)
    result['training']['history'] = 'DISABLED: only current transaction attributes; no historical transactions or counts'
    result['training']['history_k'] = 0
    result['training']['ablation'] = 'No-history control with same encoder and head as mean aggregation model'
    with open(output / f'{name}.json', 'w') as handle:
        json.dump(result, handle, indent=2)
    print('No-history ablation saved:', output / f'{name}.json', flush=True)


if __name__ == '__main__':
    main()
