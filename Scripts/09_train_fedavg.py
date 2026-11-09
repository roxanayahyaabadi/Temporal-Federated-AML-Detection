#!/usr/bin/env python3
"""Simulated cross-bank federated GNN on AMLworld HI-Small.
Partitions banks into N simulated consortia via stable CRC32 hash of sending bank ID.
No raw client transactions are aggregated: server receives model state and counts only.
Each client uses its own training edges, local unlabeled node statistics, and cumulative
local val/test topology; evaluation is performed on each client's own split data.
This is a federated simulation; simulated consortia are NOT real banks.
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

BASE=Path(__file__).resolve().parent

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path))
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_tab=load_module('aml_tabular_fl', BASE/'07_train_tabular.py')
_gnn=load_module('aml_graph_fl', BASE/'08_train_graph.py')


def bank_client_map(path,n_nodes,n_clients):
    client=np.full(n_nodes,-1,dtype=np.int32)
    with open(path,newline='') as f:
        for row in csv.DictReader(f):
            node=int(row['node_id'])
            raw_bank=row['bank_id'].encode('utf-8')
            client[node]=zlib.crc32(raw_bank)%n_clients
    assert np.all(client>=0),'Missing node bank IDs'
    return client


def subset(d, inds):
    ix=torch.as_tensor(inds,dtype=torch.long)
    return {key:(t[:,ix] if key=='edge_index' else t[ix]) for key,t in d.items()}


def local_context(ds, device):
    nx=torch.from_numpy(_tab.node_stats(ds['train'])).to(device)
    tr=ds['train']['edge_index'].to(device)
    va=ds['validation']['edge_index'].to(device)
    te=ds['test']['edge_index'].to(device)
    top={'train':tr,
         'validation':torch.cat((tr,va),dim=1),
         'test':torch.cat((tr,va,te),dim=1)}
    return nx,top


def local_training(global_state, ds, device, args):
    model=_gnn.EdgeGNN(performer=True).to(device)
    model.load_state_dict(global_state)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    x, top=local_context(ds,device)
    y=ds['train']['y'].numpy()
    pos=np.flatnonzero(y==1)
    neg=np.flatnonzero(y==0)
    if len(pos)==0 or len(neg)==0: return None
    rng=np.random.default_rng(args.seed+args.current_round*100+args.current_client)
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.positive_loss_weight,device=device))
    for _ in range(args.client_epochs):
        sel=np.concatenate([pos,rng.choice(neg,size=min(len(neg),len(pos)*args.negatives_per_positive),replace=False)])
        rng.shuffle(sel)
        ix=torch.as_tensor(sel,dtype=torch.long)
        model.train(); optim.zero_grad(set_to_none=True)
        h=model.encode_nodes(x,top['train'])
        logits=model.classify_edges(h,_gnn.batch_data(ds['train'],ix,device))
        loss=criterion(logits,ds['train']['y'][ix].to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),max_norm=2)
        optim.step()
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def eval_federated(model, client_data, device, split, batch_size):
    total=sum(len(ds[split]['y']) for ds in client_data)
    p=np.empty(total,dtype=np.float32)
    y=np.empty(total,dtype=np.int32)
    # Predictions return in arbitrary client order. For aggregate PR-AUC/F1 this is fine.
    ptr=0
    for cid,ds in enumerate(client_data):
        n=len(ds[split]['y'])
        if n==0:continue
        x,top=local_context(ds,device)
        p[ptr:ptr+n]=_gnn.eval_split(model,x,top[split],ds[split],device,batch_size)
        y[ptr:ptr+n]=ds[split]['y'].numpy().astype(np.int32)
        ptr+=n
    assert ptr==total
    return y,p


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path.home()/'Bank')
    ap.add_argument('--clients',type=int,default=6)
    ap.add_argument('--rounds',type=int,default=3)
    ap.add_argument('--client-epochs',type=int,default=1)
    ap.add_argument('--batch-size',type=int,default=50000)
    ap.add_argument('--negatives-per-positive',type=int,default=40)
    ap.add_argument('--positive-loss-weight',type=float,default=5.0)
    ap.add_argument('--lr',type=float,default=0.001)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    assert args.clients>=2
    t0=time.time()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base=args.root/'graph'/'hi_small'
    out=args.root/'results'/'hi_small'
    out.mkdir(parents=True,exist_ok=True)
    print(f'FedAvg-Performer: {args.clients} clients, {args.rounds} rounds, GPU {device}',flush=True)
    all_ds=_tab.load_splits(args.root)
    bank_map=bank_client_map(base/'node_mapping.csv',515088,args.clients)
    client_data=[]
    for cid in range(args.clients):
        ds={}
        for split,d in all_ds.items():
            src=d['edge_index'][0].numpy()
            ix=np.flatnonzero(bank_map[src]==cid)
            ds[split]=subset(d,ix)
        client_data.append(ds)
        print(f'Client {cid}: train={len(ds["train"]["y"]):,}, positive={int(ds["train"]["y"].sum())}',flush=True)
    global_model=_gnn.EdgeGNN(performer=True).to(device)
    state={k:v.detach().cpu().clone() for k,v in global_model.state_dict().items()}
    best_ap=-1
    for rnd in range(1,args.rounds+1):
        weighted=None; weight_total=0
        for cid,ds in enumerate(client_data):
            args.current_client=cid;args.current_round=rnd
            new=local_training(state,ds,device,args)
            if new is None: continue
            n=len(ds['train']['y'])
            if weighted is None:
                weighted={k:(v.double()*n if v.is_floating_point() else v.clone()) for k,v in new.items()}
            else:
                for k,v in new.items():
                    if v.is_floating_point():weighted[k]+=v.double()*n
            weight_total+=n
        if weight_total==0:raise ValueError('No clients with positive and negative training samples')
        state={k:(v.div(weight_total).to(dtype=global_model.state_dict()[k].dtype) if v.is_floating_point() else v) for k,v in weighted.items()}
        global_model.load_state_dict(state)
        yv,pv=eval_federated(global_model,client_data,device,'validation',args.batch_size)
        ap_val=average_precision_score(yv,pv)
        print(f'Round {rnd}/{args.rounds}: validation AP={ap_val:.6f}',flush=True)
        if ap_val>best_ap:
            best_ap=ap_val
            torch.save(state,out/'fedavg_sage_performer.pth')
    global_model.load_state_dict(torch.load(out/'fedavg_sage_performer.pth',weights_only=True,map_location='cpu'))
    yv,pv=eval_federated(global_model,client_data,device,'validation',args.batch_size)
    yt,pt=eval_federated(global_model,client_data,device,'test',args.batch_size)
    res=_tab.record('fedavg_sage_performer',pv,pt,yv,yt,out,time.time()-t0, np.concatenate([ds['test']['edge_timestamps'].numpy() for ds in client_data]))
    res['training']={'rounds':args.rounds,'clients':args.clients,
                     'client_epochs':args.client_epochs,'client_assignment':'CRC32(raw_sender_bank_id) % clients',
                     'aggregation':'FedAvg weighted by local training transaction count',
                     'graph_protocol':'per-client cumulative train, train+validation, all',
                     'note':'Simulated bank consortia; each client trains on its own sender-bank transactions only.'}
    with open(out/'fedavg_sage_performer.json','w') as f:json.dump(res,f,indent=2)
    print('\nFedAvg completed; results:',out,flush=True)

if __name__=='__main__':main()
