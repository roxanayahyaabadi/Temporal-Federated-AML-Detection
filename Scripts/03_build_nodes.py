import csv
from pathlib import Path

ROOT = Path.home() / "Bank"

INPUT = ROOT / "archive" / "HI-Small_Trans.csv"
OUTPUT = ROOT / "graph" / "hi_small" / "node_mapping.csv"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

account_to_id = {}
account_components = {}

transaction_count = 0


def get_node_id(bank, account, writer):
    # Exactly reproduce IBM's raw identifier concatenation.
    key = bank + account

    # Ensure concatenation cannot merge distinct bank-account pairs.
    components = (bank, account)

    if key in account_to_id:
        if account_components[key] != components:
            raise ValueError(f"Account identifier collision: {key}")
        return account_to_id[key]

    node_id = len(account_to_id)

    account_to_id[key] = node_id
    account_components[key] = components

    writer.writerow([
        node_id,
        key,
        bank,
        account
    ])

    return node_id


with open(INPUT, newline="") as source, \
     open(OUTPUT, "w", newline="") as destination:

    reader = csv.reader(source)
    writer = csv.writer(destination)

    next(reader)

    writer.writerow([
        "node_id",
        "ibm_account_key",
        "bank_id",
        "account_number"
    ])

    for row in reader:
        from_bank, from_account = row[1], row[2]
        to_bank, to_account = row[3], row[4]

        get_node_id(from_bank, from_account, writer)
        get_node_id(to_bank, to_account, writer)

        transaction_count += 1


print("Transactions processed:", transaction_count)
print("Unique graph nodes:", len(account_to_id))
print("Node mapping saved:", OUTPUT)
