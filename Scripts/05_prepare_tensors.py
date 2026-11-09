import csv
import gzip
import json
from pathlib import Path
from datetime import datetime

import torch

ROOT = Path.home() / "Bank"
GRAPH = ROOT / "graph" / "hi_small"

# Load vocabulary
with open(GRAPH / "feature_vocab.json") as f:
    vocab = json.load(f)

# Load node mapping
node_lookup = {}

with open(GRAPH / "node_mapping.csv", newline="") as f:
    for row in csv.DictReader(f):
        node_lookup[row["ibm_account_key"]] = int(row["node_id"])

print("Loaded nodes:", len(node_lookup))

# Containers
sources = []
targets = []
amounts = []
categories = []
timestamps = []
labels = []
edge_ids = []

start_time = datetime(2022, 9, 1)

# Test with 10,000 transactions
LIMIT = 10000

with gzip.open(GRAPH / "edges.csv.gz", "rt", newline="") as f:
    reader = csv.DictReader(f)

    for i, row in enumerate(reader):

        if i >= LIMIT:
            break

        sources.append(int(row["from_id"]))
        targets.append(int(row["to_id"]))

        amounts.append([
            float(row["amount_sent"]),
            float(row["amount_received"])
        ])

        categories.append([
            vocab["sent_currency"].get(row["sent_currency"], 0),
            vocab["received_currency"].get(row["received_currency"], 0),
            vocab["payment_format"].get(row["payment_format"], 0)
        ])

        t = datetime.strptime(
            row["timestamp"], "%Y/%m/%d %H:%M"
        )

        timestamps.append(
            int((t - start_time).total_seconds())
        )

        labels.append(int(row["is_laundering"]))
        edge_ids.append(int(row["edge_id"]))

# Convert to tensors
data = {
    "edge_index": torch.tensor(
        [sources, targets], dtype=torch.long
    ),
    "edge_amounts": torch.tensor(
        amounts, dtype=torch.float32
    ),
    "edge_categories": torch.tensor(
        categories, dtype=torch.long
    ),
    "edge_timestamps": torch.tensor(
        timestamps, dtype=torch.long
    ),
    "edge_ids": torch.tensor(
        edge_ids, dtype=torch.long
    ),
    "y": torch.tensor(
        labels, dtype=torch.float32
    )
}

# Validate
assert data["edge_index"].shape == (2, LIMIT)
assert data["edge_amounts"].shape == (LIMIT, 2)
assert data["edge_categories"].shape == (LIMIT, 3)
assert data["y"].shape == (LIMIT,)
assert data["edge_index"].min() >= 0
assert data["edge_index"].max() < len(node_lookup)

output = GRAPH / "sample_10000.pt"

torch.save(data, output)

print("\n=== TENSOR CONVERSION ===")

for key, value in data.items():
    print(f"{key}: {tuple(value.shape)}, {value.dtype}")

print("\nLaundering transactions:", int(data["y"].sum()))
print("Saved:", output)
