#!/usr/bin/env python3
"""Local-only baseline for 6-client Edge-History Mean AML FedAvg comparison.

Each simulated bank consortium trains its own independent model, never exchanges
weights, and only sees its locally assigned transaction history. Same client
partition, network, negative sampling, local AdamW reset per round, training
rounds, and enriched validation proxy as 13_train_temporal_fedavg.py.

For comparability with FedAvg, select ONE round for all local models based on
pooled validation-proxy AP, then choose ONE classification threshold from pooled
FULL validation predictions. Test partition is never used for selection.

This is an in-process simulation, not a privacy-preserving deployment.
"""
import argparse
import json
import random
import time
from pathlib import Path
import importlib.util

import numpy as np
import torch
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('aml_fed_reference_local', HERE / '13_train_temporal_fedavg.py')
fed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fed)
temporal = fed.temporal


def predict_local_models(states, stores, split, args, device, fixed_indices=None):
    scores, labels, times = [], [], []
    for cid, (state, store) in enumerate(zip(states, stores)):
        indices = store.group_ids[split] if fixed_indices is None else fixed_indices[cid]
        if not len(indices):
            continue
        model = temporal.TemporalEdgeNet(use_performer=False, hidden=args.hidden).to(device)
        model.load_state_dict(state)
        scores.append(temporal.predict(model, store, indices, args, device))
        offset = int(store.group_ids[split][0])
        labels.append(store.y[split][indices - offset].astype(np.int32))
        times.append(store.ts[indices].astype(np.int64))
        del model
    return np.concatenate(labels), np.concatenate(scores), np.concatenate(times)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.home() / 'Bank')
    p.add_argument('--clients', type=int, default=6)
    p.add_argument('--rounds', type=int, default=12)
    p.add_argument('--client-epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--eval-batch-size', type=int, default=2048)
    p.add_argument('--history-k', type=int, default=12)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--negatives-per-positive', type=int, default=20)
    p.add_argument('--pos-weight', type=float, default=1.0)
    p.add_argument('--lr', type=float, default=0.0005)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--validate-every', type=int, default=2)
    p.add_argument('--proxy-negatives-per-client', type=int, default=3000)
    args = p.parse_args()
    if not 2 <= args.clients <= 100:
        p.error('--clients must be between 2 and 100')
    for name in ('rounds', 'client_epochs', 'batch_size', 'eval_batch_size',
                 'history_k', 'validate_every', 'proxy_negatives_per_client',
                 'negatives_per_positive'):
        if getattr(args, name) <= 0:
            p.error(f'--{name.replace("_", "-")} must be positive')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | LOCAL-ONLY Edge-History MEAN | {args.clients} clients', flush=True)
    start = time.monotonic()
    full_splits = temporal.tab.load_splits(args.root)
    assignments = fed.assign_bank_clients(
        args.root / 'graph/hi_small/node_mapping.csv', 515088, args.clients)
    stores = fed.make_client_stores(full_splits, assignments, args.clients)
    del full_splits, assignments
    proxy_ids, proxy_y = fed.validation_proxy(
        stores, args.seed, args.proxy_negatives_per_client)
    print(f'Fixed validation proxy: {len(proxy_y):,} transactions, '
          f'{int(proxy_y.sum()):,} positives (ENRICHED; not benchmark AP)', flush=True)

    initial = temporal.TemporalEdgeNet(use_performer=False, hidden=args.hidden)
    init_state = {k: v.detach().cpu().clone() for k, v in initial.state_dict().items()}
    # Every local model starts from the same initial state used by FedAvg.
    states = [{k: v.clone() for k, v in init_state.items()} for _ in stores]
    out = args.root / 'results/hi_small'
    out.mkdir(parents=True, exist_ok=True)
    best_proxy_ap = -1.0
    checkpoint = out / 'local_only_edge_history_mean.pth'

    for rnd in range(1, args.rounds + 1):
        steps = 0
        for cid, store in enumerate(stores):
            states[cid], n_steps = fed.local_update(
                states[cid], store, args, cid, rnd, device)
            steps += n_steps
        print(f'Local-only epoch {rnd}/{args.rounds}: optimizer updates={steps}', flush=True)
        if rnd % args.validate_every == 0 or rnd == args.rounds:
            actual_y, proxy_pred, _ = predict_local_models(
                states, stores, 'validation', args, device, proxy_ids)
            if not np.array_equal(actual_y, proxy_y):
                raise AssertionError('Validation-proxy label order mismatch')
            proxy_ap = average_precision_score(proxy_y, proxy_pred)
            print(f'  validation PROXY AP={proxy_ap:.5f} '
                  '(positive-enriched, NOT full AP)', flush=True)
            if proxy_ap > best_proxy_ap:
                best_proxy_ap = float(proxy_ap)
                torch.save({'client_states': states, 'selected_round': rnd}, checkpoint)

    chosen = torch.load(checkpoint, map_location='cpu', weights_only=True)
    states = chosen['client_states']
    val_y, val_pred, _ = predict_local_models(
        states, stores, 'validation', args, device)
    test_y, test_pred, test_ts = predict_local_models(
        states, stores, 'test', args, device)
    name = 'local_only_edge_history_mean'
    result = temporal.tab.record(
        name, val_pred, test_pred, val_y, test_y, out,
        time.monotonic() - start, test_ts)
    # Keep same within-client order as scored arrays, not original CSV edge order.
    np.savez_compressed(out / f'{name}_predictions.npz',
                        validation_scores=val_pred, test_scores=test_pred)
    result['training'] = {
        'type': 'local-only, no cross-client weight sharing',
        'clients': args.clients,
        'local_epochs': args.rounds * args.client_epochs,
        'rounds': args.rounds,
        'client_epochs_per_round': args.client_epochs,
        'seed': args.seed,
        'history_k': args.history_k,
        'hidden': args.hidden,
        'lr': args.lr,
        'negatives_per_positive': args.negatives_per_positive,
        'pos_weight': args.pos_weight,
        'bank_partition': 'CRC32(normalized sending-bank ID) modulo number of clients',
        'history_scope': 'client-only, strictly earlier transaction timestamps',
        'selection': 'one shared round for all clients, maximizing pooled enriched validation proxy AP',
        'selected_round': int(chosen['selected_round']),
        'best_proxy_ap': best_proxy_ap,
        'classification_threshold': 'pooled full-validation optimal F1, fixed for test',
        'optimizer_state': 'AdamW freshly created each round, as in FedAvg baseline',
    }
    with (out / f'{name}.json').open('w') as f:
        json.dump(result, f, indent=2)
    print('Saved local-only baseline:', out / f'{name}.json', flush=True)


if __name__ == '__main__':
    main()
