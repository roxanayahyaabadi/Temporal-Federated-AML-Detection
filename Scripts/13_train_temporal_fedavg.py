#!/usr/bin/env python3
"""Simulated bank-consortium FedAvg for the causal Edge-History Mean AML model.

Compatible with 07_train_tabular.py and 11_train_temporal_edge_gnn.py in same directory.
Clients are deterministic groups of sending banks. Each client's temporal history
contains only transactions owned by that client (sender-bank partition). This
restricts cross-bank context compared with centralized training and does not
implement secure aggregation, differential privacy, or actual networked banks.

The federated server aggregates local model weights and training set sizes only;
this all-in-one-process simulation loads data centrally for repeatability.
The checkpoint is chosen on a fixed positive-enriched validation proxy; AP on
that proxy is NOT comparable to full-validation AP. Full-validation F1 determines
one global classification threshold, which is applied unchanged to the test.
"""
import argparse
import csv
import importlib.util
import json
import random
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('edge_history_impl_fed', HERE / '11_train_temporal_edge_gnn.py')
temporal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(temporal)


def assign_bank_clients(path: Path, n_nodes: int, n_clients: int) -> np.ndarray:
    assignments = np.full(n_nodes, -1, dtype=np.int16)
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            node = int(row['node_id'])
            # Normalize zero-padded bank IDs to avoid splitting the same bank.
            bank = str(int(row['bank_id']))
            client = zlib.crc32(bank.encode('utf-8')) % n_clients
            if assignments[node] != -1:
                raise ValueError(f'Node {node} assigned more than once')
            assignments[node] = client
    if (assignments < 0).any():
        raise ValueError('Incomplete node-to-bank assignments')
    return assignments


def subset_split(data: dict, indices: np.ndarray) -> dict:
    inds = torch.from_numpy(indices.astype(np.int64, copy=False))
    return {k: (v[:, inds] if k == 'edge_index' else v[inds]) for k, v in data.items()}


def make_client_stores(all_splits, assignments, n_clients: int):
    stores = []
    for cid in range(n_clients):
        client_splits = {}
        for split, data in all_splits.items():
            src = data['edge_index'][0].numpy()
            indices = np.flatnonzero(assignments[src] == cid)
            client_splits[split] = subset_split(data, indices)
        train_n = len(client_splits['train']['y'])
        npos = int(client_splits['train']['y'].sum())
        val_n = len(client_splits['validation']['y'])
        test_n = len(client_splits['test']['y'])
        print(f'Client {cid}: train={train_n:,}, positives={npos:,}, '
              f'validation={val_n:,}, test={test_n:,}', flush=True)
        if npos == 0 or npos == train_n:
            raise ValueError(f'Client {cid} lacks positive or negative training examples')
        stores.append(temporal.TemporalEdges(client_splits))
    return stores


def local_update(global_weights, store, args, cid, rnd, device):
    model = temporal.TemporalEdgeNet(use_performer=False, hidden=args.hidden).to(device)
    model.load_state_dict(global_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))
    train_y = store.y['train']
    ids = store.group_ids['train']
    positive = ids[train_y == 1]
    negative = ids[train_y == 0]
    rng = np.random.default_rng(args.seed + 1009 * rnd + 31 * cid)
    num_steps = 0
    for _ in range(args.client_epochs):
        n_neg = min(len(negative), len(positive) * args.negatives_per_positive)
        sampled = np.concatenate([positive, rng.choice(negative, size=n_neg, replace=False)])
        rng.shuffle(sampled)
        model.train()
        for start in range(0, len(sampled), args.batch_size):
            batch_ids = sampled[start:start + args.batch_size]
            data = store.batch(batch_ids, args.history_k, device)
            labels = torch.as_tensor(train_y[batch_ids].astype(np.float32), device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(data)
            loss = loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            num_steps += 1
    new_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del optimizer, model
    return new_weights, num_steps


def fedavg(updates, reference):
    """Weighted parameter average by local number of training transactions."""
    total_weight = sum(weight for weights, weight in updates)
    if total_weight <= 0:
        raise ValueError('No local updates')
    out = {}
    for key, original in reference.items():
        if original.is_floating_point():
            accumulator = torch.zeros_like(original, dtype=torch.float64)
            for weights, weight in updates:
                accumulator.add_(weights[key].to(dtype=torch.float64), alpha=float(weight))
            out[key] = (accumulator / total_weight).to(dtype=original.dtype)
        else:
            out[key] = original.clone()
    return out


def validation_proxy(stores, seed, negatives_per_client):
    rng = np.random.default_rng(seed + 919)
    ids_by_client = []
    labels = []
    for store in stores:
        va_ids = store.group_ids['validation']
        val_y = store.y['validation']
        pos = va_ids[val_y == 1]
        neg = va_ids[val_y == 0]
        neg_sample = rng.choice(neg, size=min(len(neg), negatives_per_client), replace=False)
        indices = np.concatenate([pos, neg_sample])
        ids_by_client.append(indices)
        labels.append(val_y[indices - int(va_ids[0])].astype(np.int32))
    return ids_by_client, np.concatenate(labels)


def predict_across_clients(model, stores, split, args, device, fixed_indices=None):
    predictions, labels, timestamps = [], [], []
    for cid, store in enumerate(stores):
        indices = store.group_ids[split] if fixed_indices is None else fixed_indices[cid]
        if len(indices) == 0:
            continue
        predictions.append(temporal.predict(model, store, indices, args, device))
        offset = int(store.group_ids[split][0])
        labels.append(store.y[split][indices - offset].astype(np.int32))
        timestamps.append(store.ts[indices].astype(np.int64))
    return np.concatenate(labels), np.concatenate(predictions), np.concatenate(timestamps)


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
    if not (2 <= args.clients <= 100):
        p.error('--clients must be 2..100')
    for attr in ('rounds','client_epochs','batch_size','eval_batch_size','history_k','proxy_negatives_per_client','validate_every'):
        if getattr(args, attr) <= 0:
            p.error(f'--{attr.replace("_", "-")} must be positive')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | Federated Edge-History MEAN | {args.clients} clients', flush=True)
    t0 = time.monotonic()
    full_splits = temporal.tab.load_splits(args.root)
    client_of_node = assign_bank_clients(args.root / 'graph/hi_small/node_mapping.csv', 515088, args.clients)
    stores = make_client_stores(full_splits, client_of_node, args.clients)
    del full_splits, client_of_node
    proxy_ids, proxy_y = validation_proxy(stores, args.seed, args.proxy_negatives_per_client)
    print(f'Fixed validation proxy: {len(proxy_y):,} transactions, '
          f'{proxy_y.sum():,} positives (ENRICHED; proxy AP not benchmark AP)', flush=True)
    initial = temporal.TemporalEdgeNet(use_performer=False, hidden=args.hidden)
    state = {k: v.detach().cpu().clone() for k, v in initial.state_dict().items()}
    output = args.root / 'results/hi_small'
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'fedavg_edge_history_mean.pth'
    best_proxy_ap = -1.0
    for rnd in range(1, args.rounds + 1):
        updates = []
        steps = 0
        for cid, store in enumerate(stores):
            new_weights, n_steps = local_update(state, store, args, cid, rnd, device)
            updates.append((new_weights, len(store.y['train'])))
            steps += n_steps
        state = fedavg(updates, state)
        print(f'FedAvg round {rnd}/{args.rounds}: local optimizer updates={steps}', flush=True)
        if rnd % args.validate_every == 0 or rnd == args.rounds:
            model = temporal.TemporalEdgeNet(use_performer=False, hidden=args.hidden).to(device)
            model.load_state_dict(state)
            _, proxy_pred, _ = predict_across_clients(model, stores, 'validation', args, device, proxy_ids)
            ap = average_precision_score(proxy_y, proxy_pred)
            print(f'  validation PROXY AP={ap:.5f} (positive-enriched, NOT full AP)', flush=True)
            if ap > best_proxy_ap:
                best_proxy_ap = ap
                torch.save(state, checkpoint)
            del model
    if not checkpoint.exists():
        raise RuntimeError('Missing checkpoint')
    model = temporal.TemporalEdgeNet(use_performer=False, hidden=args.hidden).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    vy, vp, _ = predict_across_clients(model, stores, 'validation', args, device)
    ty, tp, tt = predict_across_clients(model, stores, 'test', args, device)
    result = temporal.tab.record('fedavg_edge_history_mean', vp, tp, vy, ty, output,
                                 time.monotonic() - t0, tt)
    np.savez_compressed(output / 'fedavg_edge_history_mean_predictions.npz',
                        validation_scores=vp, test_scores=tp)
    result['training'] = {
        'federated_clients': args.clients,
        'rounds': args.rounds,
        'client_epochs': args.client_epochs,
        'local_optimizer_updates_each_round': 'sum across clients, printed in training log',
        'bank_partition': 'CRC32(normalized sending-bank ID) modulo number of clients',
        'history_scope': 'client-only preceding transactions, using strictly earlier timestamps',
        'aggregation': 'FedAvg weighted by client training-transaction counts',
        'history_k': args.history_k, 'seed': args.seed,
        'negatives_per_positive': args.negatives_per_positive,
        'checkpoint_selected_by': 'fixed positive-enriched validation proxy AP',
        'classification_threshold_selected_by': 'full validation F1',
        'best_proxy_ap': best_proxy_ap,
        'limitations': 'Simulated sender-bank consortia; one process holds full data; no secure aggregation/DP; cross-client history unavailable',
    }
    with (output / 'fedavg_edge_history_mean.json').open('w') as f:
        json.dump(result, f, indent=2)
    print('Saved federated temporal results:', output / 'fedavg_edge_history_mean.json', flush=True)


if __name__ == '__main__':
    main()
