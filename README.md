# Hierarchical-Demand-Forecast-using-PySpark

**Objective**
- Implement a top-down hierarchical forecasting approach on SKU-level demand data.
- Leverage **Prophet** for robust seasonality modeling and optionally **XGBoost** for improved accuracy on residuals.

**Benefits**
- **Scalable**: Built on PySpark + Pandas UDFs to handle large datasets in distributed environments.
- **Flexible**: Easily swap forecasting engines (Prophet, XGBoost, ARIMA).
- **Transparent**: Top-down allocation uses historical shares to distribute aggregate forecasts to product/location levels.

**How It Works**
1. **Hierarchy Setup**:  
   - **Top**: Total demand by date.  
   - **Mid**: Demand by location.  
   - **Bottom**: Demand by (location, SKU).  
2. **Forecasting**:  
   - Fit Prophet on Top series (12-month horizon).  
   - (Optional) Train XGBoost on top residuals to refine forecasts.  
3. **Top-Down Allocation**:  
   - Compute average historical shares at Mid and Bottom levels.  
   - Distribute aggregate forecasts proportionally.  
4. **Validation & Metrics**:  
   - Calculate MAE and RMSE on overlapping historical vs forecast.

**Usage**
```bash
$ spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2 forecast.py
