from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, sum as _sum, pandas_udf
from pyspark.sql.types import StructType, StructField, TimestampType, DoubleType
import pandas as pd
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Initialize Spark session with Arrow for efficient Pandas UDFs
def create_spark_session():
    return (SparkSession.builder
            .appName("HierarchicalForecastTopDown")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate())

spark = create_spark_session()

# 1. Load data (adjust path/columns to your Kaggle dataset)
df = (spark.read.option("header", True)
           .csv("/path/to/kaggle_demand.csv"))
df = (df.withColumn("date", to_timestamp("date", "yyyy-MM-dd"))
        .withColumn("demand", col("demand").cast("double")))

# 2. Build hierarchy: Top=date, Mid=location, Bottom=(location, sku)
bottom = df.groupBy("date","location","sku").agg(_sum("demand").alias("demand_bot"))
mid    = bottom.groupBy("date","location").agg(_sum("demand_bot").alias("demand_mid"))
top    = mid.groupBy("date").agg(_sum("demand_mid").alias("demand_top")).orderBy("date")

# 3. Forecast top series with Prophet inside Pandas UDF
schema_fc = StructType([StructField("date", TimestampType()), StructField("forecast", DoubleType())])

@pandas_udf(schema_fc)
def forecast_prophet(pdf):
    pdf = pdf.sort_values("date").rename(columns={"demand_top":"y", "date":"ds"})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(pdf)
    future = m.make_future_dataframe(periods=12, freq='M')
    fc = m.predict(future)
    return pd.DataFrame({"date": fc['ds'], "forecast": fc['yhat']})

# prepare for grouping
top2 = top.withColumn("grp", col("date")*0 + 1)
fc_top = top2.groupBy("grp").apply(forecast_prophet).drop("grp")

# 4. Combine history + forecast, compute accuracy
hist = top.withColumnRenamed("demand_top","value").withColumn("type", col("date")*0 + "history")
pred = fc_top.withColumnRenamed("forecast","value").withColumn("type", col("date")*0 + "forecast")
combined = hist.unionByName(pred).orderBy("date")

# Compute accuracy on overlapping period
hist_pd = top.toPandas().set_index('date')
fc_pd   = fc_top.toPandas().set_index('date')
common_idx = hist_pd.index.intersection(fc_pd.index)
mae = mean_absolute_error(hist_pd.loc[common_idx,'demand_top'], fc_pd.loc[common_idx,'forecast'])
mse = mean_squared_error(hist_pd.loc[common_idx,'demand_top'], fc_pd.loc[common_idx,'forecast'], squared=False)
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# 5. Top-down shares
de_converter = bottom.join(mid, ["date","location"]).withColumn("share", col("demand_bot")/col("demand_mid"))
leaf_shares = de_converter.groupBy("location","sku").mean("share").withColumnRenamed("avg(share)","avg_share")
mid_shares  = mid.join(top, "date").withColumn("mid_share", col("demand_mid")/col("demand_top")).groupBy("location").mean("mid_share").withColumnRenamed("avg(mid_share)","avg_mid_share")

# 6. Disaggregate forecasts
pred_mid    = pred.crossJoin(mid_shares).withColumn("demand_mid_fc", col("value")*col("avg_mid_share")).select("date","location","demand_mid_fc")
pred_bottom = pred_mid.join(leaf_shares, "location").withColumn("demand_bot_fc", col("demand_mid_fc")*col("avg_share")).select("date","location","sku","demand_bot_fc")

# 7. Save CSVs
combined.toPandas().to_csv("top_series_fc.csv", index=False)
pred_bottom.toPandas().to_csv("bottom_fc.csv", index=False)

# 8. Plot top-level
pdf = combined.toPandas()
plt.figure(figsize=(10,5))
hist_pd = pdf[pdf.type=="history"]
fc_pd   = pdf[pdf.type=="forecast"]
plt.plot(hist_pd.date, hist_pd.value, label="History")
plt.plot(fc_pd.date,   fc_pd.value,   label="Forecast", linestyle='--')
plt.legend()
plt.title("Top-Level Demand: History vs Forecast")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.tight_layout()
plt.savefig("top_level_demand.png")
plt.show()
