# this is link of dataset 
# https://www.kaggle.com/datasets/zanjibar/100-million-data-csv

###############################################
#              SPARK SESSION SETUP            #
###############################################

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, mean
from pyspark.sql.types import (
    StructType, StructField, IntegerType, LongType,
    StringType, FloatType, TimestampType, DoubleType
)

# Create Spark session with performance configs
spark = SparkSession.builder \
    .appName("EcommerceAnalysis") \
    .master("local[*]") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


###############################################
#                  READ DATA                  #
###############################################

# Define schema manually for performance
schema = StructType([
    StructField("event_time", TimestampType(), True),
    StructField("event_type", StringType(), True),
    StructField("product_id", IntegerType(), True),
    StructField("category_id", StringType(), True),
    StructField("category_code", StringType(), True),
    StructField("brand", StringType(), True),
    StructField("price", FloatType(), True),
    StructField("user_id", LongType(), True),
    StructField("user_session", StringType(), True)
])

# Load CSV
df = spark.read.csv(
    "file:///home/eng-mohammed/Desktop/Data.csv",
    header=True,
    schema=schema
)

df.cache()
df.printSchema()


###############################################
#      DATA UNDERSTANDING & NULL CHECKS       #
###############################################

print("Total Records:", df.count())
df.describe().show()

# Count nulls in each column
null_counts = df.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c)
    for c in df.columns
])
null_counts.show()


###############################################
#              DATA CLEANING STAGE            #
###############################################

# Replace literal "NULL" strings with actual null values
df = df.withColumn("product_id", when(col("product_id")=="NULL", None).otherwise(col("product_id"))) \
       .withColumn("brand", when(col("brand")=="NULL", None).otherwise(col("brand"))) \
       .withColumn("price", when(col("price")=="NULL", None).otherwise(col("price"))) \
       .withColumn("category_code", when(col("category_code")=="NULL", None).otherwise(col("category_code")))

# Remove exact duplicates
df = df.dropDuplicates()

# Drop rows with missing product_id (must exist)
df = df.filter(col("product_id").isNotNull())

# Fill missing price with mean price
mean_price = df.select(mean("price")).collect()[0][0]
df_clean = df.withColumn(
    "price",
    when(col("price").isNull(), mean_price).otherwise(col("price"))
)

df_clean.show(5)


###############################################
#            FEATURE ENGINEERING              #
###############################################

df_fe = (df_clean
    .withColumn("event_date", F.to_date("event_time"))
    .withColumn("event_month", F.month("event_time"))
    .withColumn("event_hour", F.hour("event_time"))
    .withColumn("category_main",
                F.when(F.col("category_code").isNotNull(),
                       F.split("category_code", "\\.")[0])
                .otherwise(None))
)

df_fe.show(5)

# Event distribution (view, cart, purchase)
df_fe.groupBy("event_type").count().show()


###############################################
#     TOP VIEWED PRODUCTS (POPULARITY)        #
###############################################

top_viewed_products = (
    df_fe
    .filter(F.col("event_type") == "view")
    .groupBy("product_id", "category_main")
    .count()
    .orderBy(F.desc("count"))
)

top_viewed_products.show(10)


###############################################
#         REVENUE CALCULATION (IMPORTANT)     #
###############################################

# Ensure price is numeric (Double)
df_fe = df_fe.withColumn(
    "price",
    F.regexp_replace("price", ",", "").cast(DoubleType())
)

# Remove duplicate events (same session + product + exact time)
df_fe = df_fe.dropDuplicates(["user_session", "product_id", "event_time"])

# Filter purchase events with valid data
df_purchase = df_fe.filter(
    (F.col("event_type") == "purchase") &
    (F.col("brand").isNotNull()) &
    (F.col("price") > 0)
)

# Revenue per brand
brand_revenue = (
    df_purchase
    .groupBy("brand")
    .agg(F.round(F.sum("price"), 2).alias("total_revenue"))
    .orderBy(F.desc("total_revenue"))
)

brand_revenue.show(10, truncate=False)
brand_revenue.printSchema()


###############################################
#            HOURLY TRAFFIC ANALYSIS          #
###############################################

# Traffic per hour
hourly_traffic = (
    df_fe
    .groupBy("event_hour")
    .count()
    .orderBy("event_hour")
)

hourly_traffic.show(24)

# Revenue per hour
hourly_revenue = (
    df_fe
    .filter(F.col("event_type") == "purchase")
    .groupBy("event_hour")
    .agg(
        F.count("*").alias("num_purchases"),
        F.round(F.sum("price"), 2).alias("total_revenue")
    )
    .orderBy("event_hour")
)

hourly_revenue.show(24)
