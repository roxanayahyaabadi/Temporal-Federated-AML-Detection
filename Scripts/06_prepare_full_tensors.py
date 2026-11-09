import csv
import gzip
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

ROOT = Path.home() / "Bank"
GRAPH = ROOT / "graph" / "hi_small"
OUTPUT = GRAPH / "tensors"

OUTPUT.mkdir(parents=True, exist_ok=True)

with open(GRAPH / "feature_vocab.json") as f:
    vocab = json.load(f)

# Counts verified by the IBM temporal split
split_sizes = {
    "train": 3248921,
    "validation": 965524,
    "test": 863900,
}

expected_positives = {
    "train": 2530,
    "validation": 1036,
    "test": 1611,
}

NUM_NODES = 515088

# Allocate CPU memory in advance
data = {}

for split, n in split_sizes.items():
    data[split] = {
        "edge_index": np.empty((2, n), dtype=np.int64),
        "edge_amounts": np.empty((n, 2), dtype=np.float32),
        "edge_categories": np.empty((n, 3), dtype=np.int64),
        "edge_timestamps": np.empty(n, dtype=np.int64),
        "edge_ids": np.empty(n, dtype=np.int64),
        "y": np.empty(n, dtype=np.float32),
    }

counts = {split: 0 for split in split_sizes}
positives = {split: 0 for split in split_sizes}

start_time = datetime(2022, 9, 1)

input_path = GRAPH / "edges.csv.gz"

print("Reading complete IBM AML graph...")

with gzip.open(input_path, "rt", newline="") as f:
    reader = csv.DictReader(f)

    for original_idx, row in enumerate(reader):

        split = row["split"]

        if split not in data:
            raise ValueError(f"Unknown split: {split}")

        i = counts[split]
        if i >= split_sizes[split]:
            raise ValueError(f"Too many rows in {split}")

        d = data[split]

        edge_id = int(row["edge_id"])
        if edge_id != original_idx:
            raise ValueError("Unexpected edge ID ordering")

        src = int(row["from_id"])
        dst = int(row["to_id"])

        if not (0 <= src < NUM_NODES and 0 <= dst < NUM_NODES):
            raise ValueError("Invalid graph node ID")

        d["edge_index"][:, i] = [src, dst]

        d["edge_amounts"][i] = [
            float(row["amount_sent"]),
            float(row["amount_received"])
        ]

        d["edge_categories"][i] = [
            vocab["sent_currency"].get(row["sent_currency"], 0),
            vocab["received_currency"].get(row["received_currency"], 0),
            vocab["payment_format"].get(row["payment_format"], 0)
        ]

        timestamp = datetime.strptime(
            row["timestamp"], "%Y/%m/%d %H:%M"
        )

        d["edge_timestamps"][i] = int(
            (timestamp - start_time).total_seconds()
        )

        label = int(row["is_laundering"])
        if label not in (0, 1):
            raise ValueError("Invalid laundering label")

        d["y"][i] = label
        d["edge_ids"][i] = edge_id

        counts[split] += 1
        positives[split] += label

# Validate and save each partition
print("\n=== FULL GRAPH TENSOR CONVERSION ===")

for split in ["train", "validation", "test"]:

    assert counts[split] == split_sizes[split]
    assert positives[split] == expected_positives[split]

    tensors = {
        key: torch.from_numpy(value)
        for key, value in data[split].items()
    }

    output_path = OUTPUT / f"{split}.pt"

    torch.save(tensors, output_path)

    print(
        f"{split}: {counts[split]:,} edges, "
        f"{positives[split]:,} laundering"
    )

    print("Saved:", output_path)

print("\nConversion completed successfully.")
