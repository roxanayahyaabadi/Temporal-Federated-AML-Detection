#!/usr/bin/env python3
"""Centralized GraphSAGE and GraphSAGE+Performer transaction classification on IBM HI-Small.
Cumulative graphs follow IBM Multi-GNN evaluation: train, train+val, train+val+test.
Node features are *training-only* unlabeled in/out counts and amount aggregates.
Loss uses all train positives plus sampled train negatives (see --negatives-per-positive).
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.attention import PerformerAttention

import importlib.util
_mod_path = Path(__file__).resolve().with_name("07_train_tabular.py")
_spec = importlib.util.spec_from_file_location("aml_tabular", _mod_path)
_tab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tab)
load_splits, node_stats, record = _tab.load_splits, _tab.node_stats, _tab.record


class EdgeGNN(nn.Module):
    def __init__(self, node_dim=4, hidden=32, performer=False):
        super().__init__()
        self.performer = performer
        self.sage1 = SAGEConv(node_dim, hidden)
        self.sage2 = SAGEConv(hidden, hidden)
        self.sent_cur = nn.Embedding(16, 8)
        self.recv_cur = nn.Embedding(16, 8)
        self.payment = nn.Embedding(8, 8)
        self.edge_proj = nn.Sequential(nn.Linear(7+24,hidden), nn.ReLU(), nn.Dropout(0.1))
        if performer:
            # Attention operates on three per-transaction tokens (sender, receiver, transaction).
            # NOT full-graph attention; scalable on 24GB GPU.
            self.attn = PerformerAttention(channels=hidden, heads=2, head_channels=16)
            self.classifier = nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
        else:
            self.classifier = nn.Sequential(nn.Linear(3*hidden,64),nn.ReLU(),nn.Dropout(0.1),nn.Linear(64,1))

    def encode_nodes(self, node_x, topology):
        h = torch.relu(self.sage1(node_x, topology))
        h = nn.functional.dropout(h,p=0.1,training=self.training)
        return torch.relu(self.sage2(h, topology))

    def classify_edges(self, h, batch):
        edge_ids, cat, a, seconds = batch
        src, dst = edge_ids[0], edge_ids[1]
        a = a.clamp(min=0)
        log_a = torch.log1p(a)
        hour = seconds.float().remainder(86400)/3600.0
        nums = torch.stack([
            log_a[:,0],log_a[:,1],(log_a[:,0]-log_a[:,1]).abs(),
            (cat[:,0]==cat[:,1]).float(),(src==dst).float(),
            torch.sin(hour*torch.pi/12),torch.cos(hour*torch.pi/12)
        ],dim=-1)
        z = torch.cat([
            nums,
            self.sent_cur(cat[:,0]),self.recv_cur(cat[:,1]),self.payment(cat[:,2])
        ],dim=-1)
        e = self.edge_proj(z)
        if self.performer:
            token = torch.stack([h[src],h[dst],e],dim=1)
            fused = self.attn(token)
            return self.classifier(fused[:,2,:]).flatten()
        return self.classifier(torch.cat([h[src],h[dst],e],dim=-1)).flatten()


def batch_data(d, indices, dev):
    # indices: CPU long tensor or slice
    return (d['edge_index'][:,indices].to(dev),
            d['edge_categories'][indices].to(dev),
            d['edge_amounts'][indices].to(dev),
            d['edge_timestamps'][indices].to(dev))


@torch.no_grad()
def eval_split(model, node_x, topology, d, device, batch_size):
    model.eval()
    h = model.encode_nodes(node_x, topology)
    p=[]
    for i in range(0,len(d['y']),batch_size):
        batch=batch_data(d,slice(i,min(i+batch_size,len(d['y']))),device)
        scores=model.classify_edges(h,batch)
        p.append(torch.sigmoid(scores).cpu().numpy())
    return np.concatenate(p)


def train_model(kind, splits, node_x, topology, args, out):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device=node_x.device
    model=EdgeGNN(performer=(kind=='sage_performer')).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    y=splits['train']['y'].numpy().astype(np.int32)
    pos=np.flatnonzero(y==1)
    neg=np.flatnonzero(y==0)
    rng=np.random.default_rng(args.seed)
    yv=splits['validation']['y'].numpy().astype(np.int32)
    yt=splits['test']['y'].numpy().astype(np.int32)
    bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.positive_loss_weight,device=device))
    best_ap=-1.0
    t0=time.time()
    print(f'\nTraining {kind}: full training graph, {len(pos)} positives + {args.negatives_per_positive} negatives per positive / epoch.',flush=True)

    for epoch in range(1,args.epochs+1):
        sample=np.concatenate([pos,rng.choice(neg,size=min(len(neg),len(pos)*args.negatives_per_positive),replace=False)])
        rng.shuffle(sample)
        idx=torch.as_tensor(sample,dtype=torch.long)
        model.train()
        opt.zero_grad(set_to_none=True)
        h=model.encode_nodes(node_x,topology['train'])
        batch=batch_data(splits['train'],idx,device)
        logits=model.classify_edges(h,batch)
        target=splits['train']['y'][idx].to(device)
        loss=bce(logits,target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),max_norm=2.)
        opt.step()
        del h,batch,logits,target
        # Monitor validation Average Precision and checkpoint; test is untouched during training.
        from sklearn.metrics import average_precision_score
        val_p=eval_split(model,node_x,topology['validation'],splits['validation'],device,args.batch_size)
        ap=float(average_precision_score(yv,val_p))
        print(f'{kind} epoch {epoch}/{args.epochs}: train_loss={loss.item():.6f}, val_AP={ap:.6f}',flush=True)
        if ap > best_ap:
            best_ap=ap
            torch.save(model.state_dict(),out/f'{kind}.pth')
        del val_p

    model.load_state_dict(torch.load(out/f'{kind}.pth',map_location=device,weights_only=True))
    pv=eval_split(model,node_x,topology['validation'],splits['validation'],device,args.batch_size)
    pt=eval_split(model,node_x,topology['test'],splits['test'],device,args.batch_size)
    result=record(kind,pv,pt,yv,yt,out,time.time()-t0,splits['test']['edge_timestamps'].numpy())
    result['training']={'epochs':args.epochs,'learning_rate':args.lr,
                        'negatives_per_positive':args.negatives_per_positive,
                        'positive_loss_weight':args.positive_loss_weight,
                        'validation_selection':'best average precision',
                        'graph_protocol':'IBM cumulative (train / train+val / train+val+test)',
                        'node_features':'train-only unlabeled account in/out counts and amounts',
                        'attention_scope':'three tokens per transaction' if kind=='sage_performer' else 'none'}
    with open(out/f"{kind}.json","w") as f:json.dump(result,f,indent=2)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path.home()/'Bank')
    ap.add_argument('--models',default='sage,sage_performer',help='sage,sage_performer')
    ap.add_argument('--epochs',type=int,default=5)
    ap.add_argument('--batch-size',type=int,default=50000)
    ap.add_argument('--negatives-per-positive',type=int,default=40)
    ap.add_argument('--positive-loss-weight',type=float,default=5.0)
    ap.add_argument('--lr',type=float,default=0.001)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda':print('WARNING: Running graph benchmarks on CPU can be slow.',flush=True)
    out=args.root/'results'/'hi_small';out.mkdir(parents=True,exist_ok=True)
    print('Loading IBM tensors, then building training-only node statistics...',flush=True)
    splits=load_splits(args.root)
    node_x=torch.from_numpy(node_stats(splits['train'])).to(device)
    tr=splits['train']['edge_index'].to(device)
    va=splits['validation']['edge_index'].to(device)
    te=splits['test']['edge_index'].to(device)
    topology={
        'train': tr,
        'validation':torch.cat([tr,va],dim=1),
        'test':torch.cat([tr,va,te],dim=1)
    }
    print('GPU:',device, '| Node feature tensor:',tuple(node_x.shape),flush=True)
    results=[]
    for kind in [s.strip() for s in args.models.split(',')]:
        if kind not in ('sage','sage_performer'):raise ValueError(kind)
        results.append(train_model(kind,splits,node_x,topology,args,out))
    print('\nCompleted graph benchmarks. Results:',out,flush=True)

if __name__=='__main__':main()
