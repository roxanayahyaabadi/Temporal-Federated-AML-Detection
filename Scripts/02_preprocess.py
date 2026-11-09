import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_DIR = Path.home() / "Bank"

DATA_PATH = PROJECT_DIR / "archive" / "HI-Small_Trans.csv"
CONFIG_PATH = PROJECT_DIR / "split_config.json"
OUTPUT_PATH = PROJECT_DIR / "processed" / "hi_small"

with open(CONFIG_PATH) as f:
    config = json.load(f)

train_end = config["train_end_exclusive"]
val_end = config["val_end_exclusive"]
test_end = config["test_end_exclusive"]

assert train_end < val_end < test_end, "Invalid split boundaries"


# --------------------------------------------------
# 2. Initialize Spark
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("IBM-AML-Preprocessing")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# --------------------------------------------------
# 3. Define schema
# --------------------------------------------------

columns = [
    "timestamp_raw",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received_raw",
    "receiving_currency",
    "amount_paid_raw",
    "payment_currency",
    "payment_format",
    "label_raw",
]

schema = StructType([
    StructField(name, StringType(), True)
    for name in columns
])


# --------------------------------------------------
# 4. Load transactions
# --------------------------------------------------

df = (
    spark.read
    .option("header", "true")
    .option("enforceSchema", "true")
    .schema(schema)
    .csv(str(DATA_PATH))
)

print("Original transaction count:", df.count())


# --------------------------------------------------
# 5. Convert data types
# --------------------------------------------------

df = (
    df
    .withColumn(
        "timestamp",
        F.to_timestamp("timestamp_raw", "yyyy/MM/dd HH:mm")
    )
    .withColumn(
        "amount_received",
        F.col("amount_received_raw").cast("double")
    )
    .withColumn(
        "amount_paid",
        F.col("amount_paid_raw").cast("double")
    )
    .withColumn(
        "is_laundering",
        F.col("label_raw").cast("int")
    )
    .drop(
        "timestamp_raw",
        "amount_received_raw",
        "amount_paid_raw",
        "label_raw"
    )
)


# --------------------------------------------------
# 6. Validate records
# --------------------------------------------------

invalid = df.filter(
    F.col("timestamp").isNull()
    | F.col("amount_received").isNull()
    | F.col("amount_paid").isNull()
    | (~F.col("is_laundering").isin(0, 1))
    | F.col("is_laundering").isNull()
    | F.col("from_bank").isNull()
    | F.col("from_account").isNull()
    | F.col("to_bank").isNull()
    | F.col("to_account").isNull()
)

invalid_count = invalid.count()

print("Invalid records:", invalid_count)

if invalid_count != 0:
    raise ValueError("Dataset contains invalid records")


# --------------------------------------------------
# 7. Construct unique account identifiers
# --------------------------------------------------

df = (
    df
    .withColumn(
        "source_node",
        F.concat_ws("::", "from_bank", "from_account")
    )
    .withColumn(
        "target_node",
        F.concat_ws("::", "to_bank", "to_account")
    )
)


# --------------------------------------------------
# 8. IBM chronological splits
# --------------------------------------------------

df = df.withColumn(
    "split",
    F.when(
        F.col("timestamp") < F.lit(train_end).cast("timestamp"),
        "train"
    )
    .when(
        F.col("timestamp") < F.lit(val_end).cast("timestamp"),
        "validation"
    )
    .when(
        F.col("timestamp") < F.lit(test_end).cast("timestamp"),
        "test"
    )
    .otherwise("out_of_range")
)

if df.filter(F.col("split") == "out_of_range").limit(1).count():
    raise ValueError("Transactions found outside configured date range")


# --------------------------------------------------
# 9. Report split statistics
# --------------------------------------------------

stats = (
    df.groupBy("split")
    .agg(
        F.count("*").alias("transactions"),
        F.sum("is_laundering").alias("laundering")
    )
    .orderBy("split")
)

stats.show(truncate=False)


# --------------------------------------------------
# 10. Export Parquet
# --------------------------------------------------

(
    df.write
    .mode("overwrite")
    .partitionBy("split")
    .parquet(str(OUTPUT_PATH))
)

print("Saved processed dataset:", OUTPUT_PATH)

spark.stop()
