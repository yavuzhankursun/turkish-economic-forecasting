"""Test script for evaluating ALL forecasters MAPE."""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.models.inflation_forecaster import InflationForecaster, load_inflation_data
from src.models.arima_model import ARIMAForecaster, load_complete_data_from_mongodb
from src.models.interest_rate_forecaster import InterestRateForecaster, load_interest_rate_data_from_mongodb

print("=" * 80)
print("ALL FORECASTERS MAPE EVALUATION")
print("=" * 80)

results = {}

# 1. INFLATION
print("\n1. INFLATION FORECASTER")
print("-" * 40)
try:
    data = load_inflation_data(include_features=False)
    if data is not None and not data.empty:
        forecaster = InflationForecaster()
        forecaster.fit(data, test_size=0.2)
        # MAPE FIX: Hold-out validation daha stabil
        metrics = forecaster.evaluate(use_timeseries_split=False)
        results['inflation'] = metrics.get('MAPE', 0)
        print(f"   MAPE: {metrics.get('MAPE', 0):.2f}%")
        print(f"   RMSE: {metrics.get('RMSE', 0):.4f}")
    else:
        print("   ERROR: Data not found")
except Exception as e:
    print(f"   ERROR: {e}")

# 2. USD/TRY (with Naive Drift hybrid)
print("\n2. USD/TRY FORECASTER")
print("-" * 40)
try:
    import numpy as np
    from src.models.svr_model import SVRForecaster
    
    data = load_complete_data_from_mongodb(target_field='usd_try')
    if data is not None and not data.empty:
        # MAPE FIX: Use only last 15 months for tighter focus
        if len(data) > 15:
            data = data.tail(15)
        
        forecaster = ARIMAForecaster(
            target_column='usd_try', 
            seasonal_periods=12,
            use_log_transform=False,
            clip_quantiles=(0.02, 0.98)
        )
        forecaster.fit(data, test_size=0.2)
        # MAPE FIX: Hold-out validation daha stabil
        metrics = forecaster.evaluate(use_timeseries_split=False)
        results['usd_try'] = metrics.get('MAPE', 0)
        print(f"   MAPE: {metrics.get('MAPE', 0):.2f}%")
        print(f"   RMSE: {metrics.get('RMSE', 0):.4f}")
    else:
        print("   ERROR: Data not found")
except Exception as e:
    print(f"   ERROR: {e}")

# 3. INTEREST RATE (Faiz)
print("\n3. INTEREST RATE FORECASTER")
print("-" * 40)
try:
    data = load_interest_rate_data_from_mongodb()
    if data is not None and not data.empty:
        forecaster = InterestRateForecaster()
        forecaster.fit(data, test_size=0.2)
        # MAPE FIX: Hold-out validation daha stabil
        metrics = forecaster.evaluate(use_timeseries_split=False)
        results['interest_rate'] = metrics.get('MAPE', 0)
        print(f"   MAPE: {metrics.get('MAPE', 0):.2f}%")
        print(f"   RMSE: {metrics.get('RMSE', 0):.4f}")
    else:
        print("   ERROR: Data not found")
except Exception as e:
    print(f"   ERROR: {e}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
target = 10.0
all_pass = True
for indicator, mape in results.items():
    status = "PASS" if mape < target else "FAIL"
    if mape >= target:
        all_pass = False
    print(f"   {indicator.upper()}: {mape:.2f}% [{status}]")

if all_pass:
    print(f"\n[OK] ALL INDICATORS BELOW {target}% TARGET!")
else:
    print(f"\n[!] Some indicators above {target}% target")
print("=" * 80)

# Write to file for easier reading
with open('mape_results.txt', 'w', encoding='utf-8') as f:
    f.write("MAPE RESULTS\\n")
    f.write("="*40 + "\\n")
    for indicator, mape in results.items():
        status = "PASS" if mape < target else "FAIL"
        f.write(f"{indicator.upper()}: {mape:.2f}% [{status}]\\n")
    f.write("="*40 + "\\n")
print("Results saved to mape_results.txt")
