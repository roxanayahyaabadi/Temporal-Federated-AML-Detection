# Temporal-Federated-AML-Detection

**Graph-based and temporal transaction-history models for anti-money laundering detection on IBM AMLworld, with centralized, simulated federated, and local-only comparisons.**

This repository is a **new research-oriented implementation** of an AML detection pipeline, not a recovery of IBM's original model code and not a production banking system. It covers transaction-level classification, graph-based baselines, causal account-history encoders, and simulated federated averaging (FedAvg).

## Research questions

1. Does knowledge of an account's previous transactions improve laundering detection beyond the current payment's attributes?
2. Does FedAvg improve detection compared with independent models when banks are partitioned into six simulated client groups?

## Dataset

We use **IBM AMLworld HI-Small**, a synthetic dataset with directed transfers between bank accounts. Each CSV row describes a payment and contains a binary `Is Laundering` label. The original CSV columns include timestamp, sender bank/account, recipient bank/account, paid and received amounts, the corresponding currencies, and payment format. The two columns named `Account` represent **different endpoints**.

Our audited copy has **5,078,345 transactions**, **5,177 positive labels (0.102%)**, and **515,088 observed graph nodes**, each identified by a compound `(bank, account)` key. A transaction is a directed edge from its sender node to its receiver node.

| Chronological partition | Dates in 2022 | Transactions | Laundering cases | Positive rate |
|---|---|---:|---:|---:|
| Training | Sep 1–6 | 3,248,921 | 2,530 | 0.0779% |
| Validation | Sep 7–8 | 965,524 | 1,036 | 0.1073% |
| Test | Sep 9–18 | 863,900 | 1,611 | 0.1865% |
| **Total** | **Sep 1–18** | **5,078,345** | **5,177** | **0.1019%** |

We split at complete-day boundaries to retain chronological order. These splits are approximately **64% / 19% / 17% by transaction count**, rather than an exact 60/20/20 split.

**Important test-set shift.** September 9–10 contains 862,792 transactions with 956 positive labels (0.111%). September 11–18 contains just 1,108 transactions, of which 655 are positive (59.1%). We therefore report both full-test and period-specific results. Statistics above describe our audited file; other versions of IBM AMLworld may differ.

**Obtain the data separately:** [IBM AML-Data repository](https://github.com/IBM/AML-Data). This repository intentionally does **not** include the raw CSVs, Parquet files, or graph tensors; check the dataset's license before any redistribution.

## Methods

**Tabular baselines.** We train class-weighted logistic regression (SGD solver), XGBoost, and XGBoost with account-level statistics derived from training edges (degree-like counts and amount totals).

**Graph baselines.** GraphSAGE learns representations from account neighborhoods. We evaluate standard GraphSAGE and a variant with Performer attention, plus an exploratory FedAvg GraphSAGE–Performer baseline. These original graph baselines have a different graph-visibility and optimization setup from the strictly historical models below; their scores must not be interpreted as a controlled proof of model-family superiority.

**Causal edge-history encoders.** For every payment, the encoder retrieves up to **12 previous incident transactions** for each endpoint. Only events with timestamps **strictly earlier** than the current transaction are eligible; past laundering labels are never given as input. Historical features include transaction amounts and categories, incoming/outgoing direction, time difference, and repeated counterparties. We compare masked **mean pooling**, **Performer attention**, and a **no-history ablation**. The history models are edge-centric neural encoders, not multi-hop GraphSAGE models.

**Distributed learning.** We assign transactions to **six simulated clients** by a deterministic hash of the sending-bank identifier. FedAvg combines locally trained parameter updates with weights proportional to the number of client training transactions. The **local-only** control uses the same partition and training budget without parameter sharing. Client histories contain only that client's transactions. Everything runs in one process: **no secure aggregation, differential privacy, or real cross-bank deployment is implemented**.

## Evaluation protocol

- Train, validation, and test partitions follow the chronological day split above.
- Training uses the positive examples plus sampled negatives (up to 20 per positive) for the temporal models.
- A **fixed positive-enriched validation proxy** selects neural checkpoints. Its AP **is not** the benchmark AP.
- The classification threshold is selected by **maximum F1 on the complete validation partition**, then held fixed on test.
- We report **average precision (AP)**, precision, recall, and F1. AP measures ranking quality across thresholds; F1 is evaluated at the validation-selected threshold.
- Temporal predictions may use unlabeled prior activity, including earlier events within validation/test, but **never events at the same or later timestamp**.
- The causal-history evaluation is **not an exact reproduction** of IBM's reference cumulative-graph benchmark.

## Results

### Complete test partition — seed 42

| Model | Test AP | Precision | Recall | Test F1 |
|---|---:|---:|---:|---:|
| Logistic regression | 0.0080 | 0.0083 | 0.9361 | 0.0164 |
| XGBoost | 0.1070 | 0.1633 | 0.2557 | 0.1993 |
| XGBoost + graph statistics | 0.1097 | 0.1541 | 0.2427 | 0.1885 |
| GraphSAGE | 0.0027 | 0.0034 | 0.3371 | 0.0068 |
| GraphSAGE + Performer | 0.0029 | 0.0041 | 0.2048 | 0.0080 |
| FedAvg GraphSAGE + Performer | 0.0045 | 0.0069 | 0.0137 | 0.0092 |
| No-history edge model | 0.0683 | 0.0805 | 0.2148 | 0.1171 |
| **Temporal history — mean** | **0.4893** | **0.5828** | **0.5090** | **0.5434** |
| Temporal history — Performer | 0.4418 | 0.4688 | 0.4811 | 0.4749 |
| FedAvg temporal mean | 0.1410 | 0.2161 | 0.2334 | 0.2244 |
| Local-only temporal mean | 0.1472 | 0.2511 | 0.1831 | 0.2118 |

![Overall test AP and F1](figures/figure_1_full_test_AP_F1.png)

### Less atypical test period — September 9–10

| Model | Test AP | Test F1 |
|---|---:|---:|
| Temporal history — mean | **0.2792** | **0.3878** |
| Temporal history — Performer | 0.2294 | 0.3182 |
| FedAvg temporal mean | 0.0683 | 0.1439 |
| Local-only temporal mean | 0.0780 | 0.1573 |

![F1 by test period](figures/figure_2_period_specific_F1.png)

On September 9–10, FedAvg raises recall relative to local-only training (0.1872 versus 0.1632), but increases false positives (1,352 versus 871) and reduces AP and F1. Thus **FedAvg does not consistently outperform local-only training** in this simulation.

### Centralized mean model — three independent seeds

| Seed | Validation AP | Test AP | Test F1 |
|---|---:|---:|---:|
| 42 | 0.3417 | 0.4893 | 0.5434 |
| 123 | 0.3500 | 0.4623 | 0.5382 |
| 456 | 0.3662 | 0.5014 | 0.5473 |
| **Mean ± sample standard deviation** | **0.3526 ± 0.0125** | **0.4843 ± 0.0200** | **0.5430 ± 0.0046** |

Three-seed experiments for the distributed variants and history-length ablations remain pending. Treat the comparisons between the model families as exploratory because training budgets and available graph context are not identical.

## Repository structure

```text
Temporal-Federated-AML-Detection/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── 01_audit.py ... 06_prepare_full_tensors.py   # copy from local ~/Bank
│   ├── 07_train_tabular.py
│   ├── 08_train_graph.py
│   ├── 09_train_fedavg.py
│   ├── 10_make_report.py
│   ├── 11_train_temporal_edge_gnn.py
│   ├── 12_train_no_history.py
│   ├── 13_train_temporal_fedavg.py
│   └── 14_train_local_only_temporal.py
├── figures/                                      # report plots (small PNGs)
├── paper/                                        # IEEE conference-style draft
├── metrics/hi_small/                             # small result JSON files only
├── config/                                       # optional split/vocabulary metadata
└── docs/                                         # supporting method documentation
```

**Required local inputs (not in Git):** keep the downloaded dataset and generated tensor files under `~/Bank/` or another directory passed to `--root`. The scripts currently load graph inputs from `<root>/graph/hi_small/tensors/{train,validation,test}.pt` and write outputs to `<root>/results/hi_small/`. The initial data-preparation scripts (`01`–`06`) are part of the original working directory and must be copied to `scripts/` before the repository can reproduce preprocessing end to end.

## Environment

Developed on Ubuntu with **Python 3.11**, **Java 17**, **PySpark 3.5.4**, **PyTorch 2.11.0+cu128**, and **PyTorch Geometric 2.8.0**, using an NVIDIA RTX 4090. Other PyTorch/CUDA installations may require platform-specific instructions.

```bash
conda create -n federated-aml python=3.11 openjdk=17 -c conda-forge -y
conda activate federated-aml
# Install a compatible GPU-enabled PyTorch version for your environment first.
pip install -r requirements.txt
```

`requirements.txt` specifies direct dependencies; it is **not a lockfile**. To record your exact environment on the training machine, run `python -m pip freeze > requirements-lock.txt` and review the output before committing it.

## Reproduce the experiments

Set `--root` to the **data and results directory**, not necessarily the GitHub repository directory. The examples below assume your existing prepared dataset remains under `~/Bank`.

```bash
# From a clone of this GitHub repository
conda activate federated-aml
DATA_ROOT="$HOME/Bank"

# After completing dataset preparation (scripts 01–06):
python scripts/07_train_tabular.py --root "$DATA_ROOT" --models lr,xgb,xgb_graph
python scripts/08_train_graph.py --root "$DATA_ROOT" --models sage,sage_performer
python scripts/09_train_fedavg.py --root "$DATA_ROOT" --clients 6

# Centralized temporal models
python scripts/11_train_temporal_edge_gnn.py --root "$DATA_ROOT" \
    --models edge_history_mean,edge_history_performer --epochs 12 --history-k 12 --seed 42

# No-history ablation
python scripts/12_train_no_history.py --root "$DATA_ROOT" --epochs 12 --seed 42

# Six-client distributed comparison
python scripts/13_train_temporal_fedavg.py --root "$DATA_ROOT" \
    --clients 6 --rounds 12 --client-epochs 1 --history-k 12 --seed 42
python scripts/14_train_local_only_temporal.py --root "$DATA_ROOT" \
    --clients 6 --rounds 12 --client-epochs 1 --history-k 12 --seed 42
```

**Checkpoint caution:** scripts reuse their model-specific output filenames. Before another random-seed run, copy the corresponding JSON, checkpoint, and predictions into a seed-specific **local** backup directory. The GitHub `metrics/` folder should contain only the small JSON summaries, not the model checkpoints or large predictions.

## Paper and figures

The `paper/` directory contains an **IEEE four-page conference-style draft**. It is a working manuscript, not a peer-reviewed publication; update it with your latest edits and verify all references before submission. The `figures/` directory contains PNG exports of the experimental comparisons. Figures in the LaTeX draft are produced directly using `pgfplots`.

## Limitations

This study uses synthetic transactions and six simulated groups of banks. The most important limitations are the unusual prevalence shift late in the test set, unequal optimization budgets between the first graph baselines and later temporal models, limited repeated-seed results for distributed learning, and the lack of secure aggregation or differential privacy. Neither the centralized–federated gap nor the current client simulation isolates the effect of model-weight sharing from differences in history availability.

## References and attribution

- IBM AMLworld data and source material: [IBM/AML-Data](https://github.com/IBM/AML-Data).
- Altman *et al.*, “Realistic Synthetic Financial Transactions for Anti-Money Laundering Models,” NeurIPS Datasets and Benchmarks, 2023.
- Hamilton, Ying, and Leskovec, “Inductive Representation Learning on Large Graphs,” NeurIPS, 2017.
- Chen and Guestrin, “XGBoost: A Scalable Tree Boosting System,” KDD, 2016.
- Choromanski *et al.*, “Rethinking Attention with Performers,” ICLR, 2021.
- McMahan *et al.*, “Communication-Efficient Learning of Deep Networks from Decentralized Data,” AISTATS, 2017.

The IBM dataset has its own distribution terms, separate from this implementation. Review the relevant licenses before publishing data or adopting a code license for this repository.
