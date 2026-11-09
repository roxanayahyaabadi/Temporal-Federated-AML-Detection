#!/usr/bin/env python3
"""Stage 1: verify and audit IBM AMLworld transaction files via PySpark.

Usage:
  spark-submit --master 'local[*]' --driver-memory 6g scripts/01_audit.py \
      --transactions ~/Bank/archive/HI-Small_Trans.csv \
      --output outputs/audit/hi_small.json

Raw transaction data is read-only; the script writes only aggregated results.
"""

import argparse
import csv
import json
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

# The source has two distinct columns called "Account". They are the sender's
# and receiver's account numbers; map by *position* rather than by raw names.
EXPECTED_HEADER = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account",
    "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
    "Payment Format", "Is Laundering",
]

SCHEMA = StructType([
    StructField("timestamp_raw", StringType(), True),
    StructField("from_bank", StringType(), True),
    StructField("from_account", StringType(), True),
    StructField("to_bank", StringType(), True),
    StructField("to_account", StringType(), True),
    StructField("amount_received", DoubleType(), True),
    StructField("receiving_currency", StringType(), True),
    StructField("amount_paid", DoubleType(), True),
    StructField("payment_currency", StringType(), True),
    StructField("payment_format", StringType(), True),
    StructField("is_laundering", IntegerType(), True),
])


def verify_header(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        actual = next(csv.reader(handle))
    if actual != EXPECTED_HEADER:
        raise ValueError(
            f"Unexpected IBM AML header in {path}:\n"
            f"Actual:   {actual!r}\nExpected: {EXPECTED_HEADER!r}\n"
            "Check the dataset version before mapping columns by position."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/audit/hi_small.json"))
    args = parser.parse_args()

    path = args.transactions.expanduser().resolve()
    if not path.is_file():
        parser.error(f"Transaction file does not exist: {path}")
    verify_header(path)

    spark = (
        SparkSession.builder.appName("federated-aml-audit")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        raw = (
            spark.read.option("header", "true")
            .option("enforceSchema", "true")  # use positionally assigned unique names
            .option("mode", "FAILFAST")
            .schema(SCHEMA).csv(str(path))
        )
        df = (
            raw.withColumn("event_time", F.to_timestamp("timestamp_raw", "yyyy/MM/dd HH:mm"))
            .withColumn("from_node", F.concat_ws("::", "from_bank", "from_account"))
            .withColumn("to_node", F.concat_ws("::", "to_bank", "to_account"))
        )

        agg = df.agg(
            F.count("*").alias("transaction_count"),
            F.sum(F.when(F.col("is_laundering") == 1, 1).otherwise(0)).alias("laundering_count"),
            F.sum(F.when(F.col("is_laundering") == 0, 1).otherwise(0)).alias("non_laundering_count"),
            F.sum(F.when(~F.col("is_laundering").isin(0, 1) | F.col("is_laundering").isNull(), 1).otherwise(0)).alias("invalid_label_count"),
            F.sum(F.when(F.col("event_time").isNull(), 1).otherwise(0)).alias("invalid_timestamp_count"),
            F.sum(F.when(F.col("amount_paid").isNull() | F.isnan("amount_paid"), 1).otherwise(0)).alias("invalid_paid_amount_count"),
            F.sum(F.when(F.col("amount_received").isNull() | F.isnan("amount_received"), 1).otherwise(0)).alias("invalid_received_amount_count"),
            F.sum(F.when(F.col("from_account").isNull() | F.col("to_account").isNull() | F.col("from_bank").isNull() | F.col("to_bank").isNull(), 1).otherwise(0)).alias("missing_endpoint_count"),
            F.sum(F.when(F.col("from_bank") != F.col("to_bank"), 1).otherwise(0)).alias("cross_bank_count"),
            F.sum(F.when((F.col("from_bank") == F.col("to_bank")) & (F.col("from_account") == F.col("to_account")), 1).otherwise(0)).alias("self_transfer_count"),
            F.approx_count_distinct("from_node").alias("approx_unique_senders"),
            F.approx_count_distinct("to_node").alias("approx_unique_receivers"),
            F.approx_count_distinct("from_bank").alias("approx_unique_sending_banks"),
            F.approx_count_distinct("to_bank").alias("approx_unique_receiving_banks"),
            F.min("event_time").cast("string").alias("earliest_timestamp"),
            F.max("event_time").cast("string").alias("latest_timestamp"),
        ).first().asDict(recursive=True)
        agg["positive_rate"] = (agg["laundering_count"] / agg["transaction_count"] if agg["transaction_count"] else None)

        daily_rows = (
            df.withColumn("date", F.to_date("event_time"))
              .groupBy("date")
              .agg(F.count("*").alias("transactions"),
                   F.sum(F.when(F.col("is_laundering") == 1, 1).otherwise(0)).alias("laundering"))
              .orderBy("date").collect()
        )
        daily = [
            {"date": str(r["date"]), "transactions": int(r["transactions"]),
             "laundering": int(r["laundering"] or 0)}
            for r in daily_rows
        ]
        report = {
            "source_path": str(path),
            "source_file_size_bytes": path.stat().st_size,
            "data_split": "not yet performed; chronologically partition AFTER schema/label audit",
            "summary": agg,
            "counts_by_date": daily,
        }
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nSaved aggregated audit report: {output}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
