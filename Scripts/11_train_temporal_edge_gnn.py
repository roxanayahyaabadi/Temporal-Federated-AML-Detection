#!/usr/bin/env python3
"""Train two scalable edge-aware AML models using strictly earlier transaction context.

This is a *causal-history extension*, NOT an exact reproduction of IBM Multi-GNN's
cumulative, transductive graph protocol. Node context: up to K strictly earlier
incident transactions for each endpoint; context never contains laundering labels.
All positive training examples + negatives sampled each epoch; true full validation
is used only to select the decision threshold after checkpoint selection.
"""
import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score
from torch_geometric.nn.attention import PerformerAttention

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('aml_tabular_temporal', HERE / '07_train_tabular.py')
tab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tab)


class TemporalEdges:
    """CPU event store and account-indexed history. Chronology is strictly earlier.

    The account index contains ALL unlabeled events; lookup uses timestamp < query
    timestamp, so future events/labels cannot enter a prediction's history.
    Concurrent events with identical timestamps are excluded from one another.
    """

    def __init__(self, splits, n_nodes=515088):
        self.group_ids = {}
        offset = 0
        for name in ('train', 'validation', 'test'):
            n = len(splits[name]['y'])
            self.group_ids[name] = np.arange(offset, offset + n, dtype=np.int32)
            offset += n
        self.size = offset
        self.src = np.concatenate([splits[s]['edge_index'][0].numpy() for s in self.group_ids]).astype(np.int32)
        self.dst = np.concatenate([splits[s]['edge_index'][1].numpy() for s in self.group_ids]).astype(np.int32)
        self.ts = np.concatenate([splits[s]['edge_timestamps'].numpy() for s in self.group_ids]).astype(np.int32)
        self.cat = np.concatenate([splits[s]['edge_categories'].numpy() for s in self.group_ids]).astype(np.uint8)
        amount = np.concatenate([splits[s]['edge_amounts'].numpy() for s in self.group_ids])
        sent = np.log1p(np.maximum(amount[:, 0], 0.0))
        received = np.log1p(np.maximum(amount[:, 1], 0.0))
        hour = (self.ts % 86400) / 3600.0
        self.num = np.column_stack([
            sent / 10.0, received / 10.0, np.abs(sent - received) / 10.0,
            (self.cat[:, 0] == self.cat[:, 1]).astype(np.float32),
            (self.src == self.dst).astype(np.float32),
            np.sin(np.pi * hour / 12.0), np.cos(np.pi * hour / 12.0),
        ]).astype(np.float32)
        del amount, sent, received
        self.y = {k: splits[k]['y'].numpy().astype(np.int8) for k in self.group_ids}

        # Each transaction is indexed twice: outgoing at sender, incoming at receiver.
        # Encode (node, timestamp) in one int64 for efficient vectorized binary search.
        max_time = int(np.max(self.ts))
        self.stride = 1 << max(1, max_time.bit_length())
        node = np.concatenate((self.src, self.dst))
        incident_time = np.tile(self.ts, 2)
        edge_id = np.tile(np.arange(self.size, dtype=np.int32), 2)
        role = np.concatenate((np.zeros(self.size, np.uint8), np.ones(self.size, np.uint8)))
        code = node.astype(np.int64) * self.stride + incident_time
        order = np.argsort(code, kind='stable')
        self.code = code[order]
        self.edge_id = edge_id[order]
        self.role = role[order]
        self.indptr = np.empty(n_nodes + 1, np.int64)
        self.indptr[0] = 0
        np.cumsum(np.bincount(node, minlength=n_nodes), out=self.indptr[1:])
        del code, order, edge_id, incident_time, node, role
        assert self.indptr[-1] == 2 * self.size
        print(f'Built history index: {self.size:,} transactions, {n_nodes:,} accounts', flush=True)

    def batch(self, ids, k, device):
        """Return one minibatch with k most recent strictly older incident edges."""
        ids = np.asarray(ids, dtype=np.int64)
        b = len(ids)
        src = self.src[ids]
        dst = self.dst[ids]
        anchor = np.column_stack((src, dst)).reshape(-1)
        counterpart = np.column_stack((dst, src)).reshape(-1)
        qtime = np.repeat(self.ts[ids], 2)
        lo = self.indptr[anchor]
        # left bound excludes all events at the exact same timestamp.
        hi = np.searchsorted(self.code, anchor.astype(np.int64) * self.stride + qtime, side='left')
        count = hi - lo
        rank = np.arange(k, dtype=np.int64)[None, :]
        valid = rank < count[:, None]
        positions = np.maximum(hi[:, None] - 1 - rank, lo[:, None])
        positions = np.minimum(positions, len(self.edge_id) - 1)
        hid = self.edge_id[positions]
        direction = self.role[positions]
        other = np.where(direction == 0, self.dst[hid], self.src[hid])
        same_party = (other == counterpart[:, None]).astype(np.float32)
        delta_minutes = np.maximum(0, qtime[:, None].astype(np.int64) - self.ts[hid]) / 60.0
        extra = np.stack((direction.astype(np.float32), np.log1p(delta_minutes).astype(np.float32) / 12.0, same_party), axis=-1)
        extra[~valid] = 0.0
        # shape of history: [batch, 2 endpoints, k neighbors, features]
        def tensor(x, dtype):
            return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=device)
        return (
            tensor(self.num[ids], torch.float32),
            tensor(self.cat[ids].astype(np.int64), torch.long),
            tensor(self.num[hid].reshape(b, 2, k, 7), torch.float32),
            tensor(self.cat[hid].reshape(b, 2, k, 3).astype(np.int64), torch.long),
            tensor(extra.reshape(b, 2, k, 3), torch.float32),
            tensor(valid.reshape(b, 2, k), torch.bool),
            tensor((np.log1p(count.astype(np.float32)) / 10.0).reshape(b, 2), torch.float32),
        )


class TemporalEdgeNet(nn.Module):
    def __init__(self, use_performer, hidden=64, dropout=0.15):
        super().__init__()
        self.use_performer = use_performer
        self.c1 = nn.Embedding(16, 8)
        self.c2 = nn.Embedding(16, 8)
        self.fmt = nn.Embedding(8, 8)
        self.edge_encoder = nn.Sequential(nn.Linear(7 + 24, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.context_encoder = nn.Linear(3, hidden)
        self.account_proj = nn.Sequential(nn.Linear(hidden * 2 + 1, hidden), nn.ReLU(), nn.Dropout(dropout))
        if use_performer:
            self.attention = PerformerAttention(channels=hidden, heads=4, head_channels=hidden // 4, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden * 4 + 2, hidden * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, 1))

    def embed(self, nums, cats):
        z = torch.cat((nums, self.c1(cats[..., 0]), self.c2(cats[..., 1]), self.fmt(cats[..., 2])), dim=-1)
        return self.edge_encoder(z)

    def forward(self, batch):
        current_n, current_c, past_n, past_c, past_extra, mask, history_counts = batch
        b, _, k = mask.shape
        q = self.embed(current_n, current_c)
        past = self.embed(past_n, past_c) + self.context_encoder(past_extra)
        if self.use_performer:
            tokens = torch.cat((q.unsqueeze(1).expand(b, 2, -1).reshape(b * 2, 1, -1),
                                past.reshape(b * 2, k, -1)), dim=1)
            token_mask = torch.cat((torch.ones(b * 2, 1, device=mask.device, dtype=torch.bool),
                                    mask.reshape(b * 2, k)), dim=1)
            account = self.attention(tokens, mask=token_mask)[:, 0].reshape(b, 2, -1)
        else:
            weighted = past * mask.unsqueeze(-1)
            mean = weighted.sum(dim=2) / mask.sum(dim=2).clamp(min=1).unsqueeze(-1)
            account = self.account_proj(torch.cat((q.unsqueeze(1).expand(-1, 2, -1), mean, history_counts.unsqueeze(-1)), dim=-1))
        a, z = account[:, 0], account[:, 1]
        joined = torch.cat((q, a, z, (a - z).abs(), history_counts), dim=-1)
        return self.head(joined).squeeze(-1)


@torch.no_grad()
def predict(model, store, indices, args, device):
    model.eval()
    result = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), args.eval_batch_size):
        end = min(start + args.eval_batch_size, len(indices))
        batch = store.batch(indices[start:end], args.history_k, device)
        result[start:end] = torch.sigmoid(model(batch)).cpu().numpy()
    return result


def train_one(kind, store, args, device, out):
    use_performer = kind == 'edge_history_performer'
    model = TemporalEdgeNet(use_performer, hidden=args.hidden).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))
    train_y = store.y['train']; val_y = store.y['validation']; test_y = store.y['test']
    tr_ids = store.group_ids['train']; va_ids = store.group_ids['validation']; te_ids = store.group_ids['test']
    pos = tr_ids[train_y == 1]; neg = tr_ids[train_y == 0]
    rng = np.random.default_rng(args.seed)
    # Fixed proxy-validation set: ALL validation positives plus sampled negatives.
    # Proxy AP is used ONLY for checkpoint selection; full validation selects threshold.
    val_pos = va_ids[val_y == 1]; val_neg = va_ids[val_y == 0]
    sample_neg = rng.choice(val_neg, size=min(args.val_proxy_negatives, len(val_neg)), replace=False)
    proxy = np.concatenate((val_pos, sample_neg))
    proxy_y = val_y[proxy - int(va_ids[0])]
    best_ap = -1.0
    checkpoint = out / f'{kind}.pth'
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        sampled_neg = rng.choice(neg, size=min(len(neg), len(pos) * args.negatives_per_positive), replace=False)
        samples = np.concatenate((pos, sampled_neg))
        rng.shuffle(samples)
        model.train()
        total_loss = 0.0
        for lo in range(0, len(samples), args.batch_size):
            ids = samples[lo:lo + args.batch_size]
            batch = store.batch(ids, args.history_k, device)
            labels = torch.as_tensor(train_y[ids].astype(np.float32), device=device)
            optim.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optim.step()
            total_loss += loss.item() * len(ids)
        print(f'{kind} epoch {epoch}/{args.epochs}, steps={int(np.ceil(len(samples) / args.batch_size))}, loss={total_loss / len(samples):.5f}', flush=True)
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            pv = predict(model, store, proxy, args, device)
            ap_proxy = float(average_precision_score(proxy_y, pv))
            print(f'  proxy-validation AP={ap_proxy:.5f} (positive-enriched, NOT full AP)', flush=True)
            if ap_proxy > best_ap:
                best_ap = ap_proxy
                torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
    val_pred = predict(model, store, va_ids, args, device)
    test_pred = predict(model, store, te_ids, args, device)
    result = tab.record(kind, val_pred, test_pred, val_y, test_y, out,
                        time.time() - t0, store.ts[te_ids])
    np.savez_compressed(out / f'{kind}_predictions.npz',
                        validation_scores=val_pred, test_scores=test_pred)
    result['training'] = {
        'epochs': args.epochs, 'updates_per_epoch': int(np.ceil(len(samples)/args.batch_size)),
        'history_k': args.history_k, 'history': 'K most recent incident edges with timestamps strictly earlier',
        'full_validation_for_threshold': True, 'proxy_validation_AP_for_checkpoint': best_ap,
        'note': 'Causal edge-history model, not IBM cumulative snapshot GNN benchmark',
        'negatives_per_positive': args.negatives_per_positive, 'positive_loss_weight': args.pos_weight,
        'hidden': args.hidden, 'seed': args.seed,
    }
    with open(out / f'{kind}.json', 'w') as f:
        json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path.home() / 'Bank')
    ap.add_argument('--models', default='edge_history_mean,edge_history_performer')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--eval-batch-size', type=int, default=2048)
    ap.add_argument('--history-k', type=int, default=12)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--negatives-per-positive', type=int, default=20)
    ap.add_argument('--pos-weight', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=0.0005)
    ap.add_argument('--val-proxy-negatives', type=int, default=100000)
    ap.add_argument('--validate-every', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('Loading graph tensors and constructing strictly historical adjacency...', flush=True)
    store = TemporalEdges(tab.load_splits(args.root))
    out = args.root / 'results' / 'hi_small'
    out.mkdir(parents=True, exist_ok=True)
    for kind in [s.strip() for s in args.models.split(',')]:
        if kind not in ('edge_history_mean', 'edge_history_performer'):
            raise ValueError('Unknown model ' + kind)
        train_one(kind, store, args, device, out)
    print('Finished. Results:', out, flush=True)


if __name__ == '__main__':
    main()
