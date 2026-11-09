import csv
import gzip
import json
from pathlib import Path
from collections import Counter
from datetime import datetime

ROOT = Path.home() / "Bank"

TRANSACTIONS = ROOT / "archive/HI-Small_Trans.csv"
NODES = ROOT / "graph/hi_small/node_mapping.csv"
OUTPUT = ROOT / "graph/hi_small/edges.csv.gz"
CONFIG = ROOT / "split_config.json"

with open(CONFIG) as f:
    config = json.load(f)

def csv_timestamp(iso_timestamp):
    return datetime.fromisoformat(iso_timestamp).strftime(
        "%Y/%m/%d %H:%M"
    )

train_end = csv_timestamp(config["train_end_exclusive"])
val_end = csv_timestamp(config["val_end_exclusive"])
test_end = csv_timestamp(config["test_end_exclusive"])

# Load IBM-compatible node IDs
node_lookup = {}

with open(NODES, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        node_lookup[row["ibm_account_key"]] = int(row["node_id"])

print("Loaded graph nodes:", len(node_lookup))

split_counts = Counter()
positive_counts = Counter()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

expected_header = [
    "Timestamp", "From Bank", "Account", "To Bank",
    "Account", "Amount Received", "Receiving Currency",
    "Amount Paid", "Payment Currency", "Payment Format",
    "Is Laundering"
]

with open(TRANSACTIONS, newline="") as source, \
     gzip.open(OUTPUT, "wt", newline="", compresslevel=3) as destination:

    reader = csv.reader(source)
    writer = csv.writer(destination)

    if next(reader) != expected_header:
        raise ValueError("Unexpected IBM transaction CSV header")

    writer.writerow([
        "edge_id",
        "from_id",
        "to_id",
        "timestamp",
        "amount_sent",
        "sent_currency",
        "amount_received",
        "received_currency",
        "payment_format",
        "is_laundering",
        "split"
    ])

    for edge_id, row in enumerate(reader):

        if len(row) != 11:
            raise ValueError(f"Invalid transaction row: {edge_id}")

        timestamp = row[0]

        from_key = row[1] + row[2]
        to_key = row[3] + row[4]

        from_id = node_lookup[from_key]
        to_id = node_lookup[to_key]

        label = int(row[10])

        if label not in (0, 1):
            raise ValueError(f"Invalid label at edge {edge_id}")

        if timestamp < train_end:
            split = "train"
        elif timestamp < val_end:
            split = "validation"
        elif timestamp < test_end:
            split = "test"
        else:
            raise ValueError(f"Timestamp outside split range: {timestamp}")

        writer.writerow([
            edge_id,
            from_id,
            to_id,
            timestamp,
            row[7],
            row[8],
            row[5],
            row[6],
            row[9],
            label,
            split
        ])

        split_counts[split] += 1
        positive_counts[split] += label

print("\n=== GRAPH CONSTRUCTION COMPLETE ===")

print("Total edges:", sum(split_counts.values()))
print("Unique nodes:", len(node_lookup))

for split in ["train", "validation", "test"]:
    print(
        f"{split}: "
        f"{split_counts[split]} edges, "
        f"{positive_counts[split]} laundering"
    )

print("\nSaved graph edges:", OUTPUT)
