"""
TUBITAK Raporu - Tablo 5: Model Performans Karsilastirmasi (12 Aylik Tahmin)
ARIMA, SVR ve Hibrit (ARIMA+SVR) modelleri icin RMSE, MAPE, MAE hesaplar.
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

STEPS = 12  # 12 aylik tahmin

def metrics(actual: np.ndarray, pred: np.ndarray):
    """RMSE, MAPE (%), MAE hesapla."""
    actual = np.asarray(actual).ravel()
    pred = np.asarray(pred).ravel()
    n = min(len(actual), len(pred))
    actual, pred = actual[:n], pred[:n]
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    denom = np.abs(actual) + 1e-10
    mape = np.mean(np.abs((actual - pred) / denom)) * 100
    return {'RMSE': round(rmse, 2), 'MAPE': round(mape, 2), 'MAE': round(mae, 2)}


def run_indicator(name, train, test, arima_forecaster_class, use_hybrid_forecast=True,
                  svr_C=100.0, svr_epsilon=0.1, lag=6):
    """Bir gosterge icin ARIMA, SVR, Hibrit metriklerini hesapla."""
    from src.models.svr_model import SVRForecaster

    actual = test.values[:STEPS]
    steps = min(STEPS, len(test))
    arima_metrics = None
    arima_pred = None
    svr_metrics = None
    svr_pred = None
    fc = None

    # ARIMA
    try:
        fc = arima_forecaster_class()
        fc.fit(train, test_size=0.0)
        try:
            res = fc.forecast(steps=steps, use_hybrid=False)
        except TypeError:
            res = fc.forecast(steps=steps)
        if res and isinstance(res, dict) and 'forecast' in res:
            arima_pred = np.asarray(res['forecast'].values)[:steps]
            arima_metrics = metrics(actual[:len(arima_pred)], arima_pred)
    except Exception as e:
        print(f"  ARIMA hata: {e}")

    # SVR
    try:
        lag_use = min(lag, max(3, len(train) // 2 - 1))
        svr = SVRForecaster(kernel='rbf', C=svr_C, epsilon=svr_epsilon)
        svr.fit(train, lag=lag_use)
        svr_pred_series = svr.predict(train, steps=steps)
        svr_pred = np.asarray(svr_pred_series.values)[:steps]
        svr_metrics = metrics(actual[:len(svr_pred)], svr_pred)
    except Exception as e:
        print(f"  SVR hata: {e}")

    # Hibrit
    hybrid_metrics = None
    try:
        if use_hybrid_forecast and fc is not None and hasattr(fc, 'forecast'):
            import inspect
            sig = inspect.signature(fc.forecast)
            if 'use_hybrid' in sig.parameters:
                res = fc.forecast(steps=steps, use_hybrid=True)
                if res and 'forecast' in res:
                    hy_pred = np.asarray(res['forecast'].values)[:steps]
                    hybrid_metrics = metrics(actual[:len(hy_pred)], hy_pred)
        if hybrid_metrics is None and arima_pred is not None and svr_pred is not None:
            n = min(len(arima_pred), len(svr_pred), len(actual))
            hy_pred = 0.6 * arima_pred[:n] + 0.4 * svr_pred[:n]
            hybrid_metrics = metrics(actual[:n], hy_pred)
    except Exception as e:
        if arima_pred is not None and svr_pred is not None:
            n = min(len(arima_pred), len(svr_pred), len(actual))
            hy_pred = 0.6 * arima_pred[:n] + 0.4 * svr_pred[:n]
            hybrid_metrics = metrics(actual[:n], hy_pred)

    return {'ARIMA': arima_metrics, 'SVR': svr_metrics, 'Hibrit': hybrid_metrics}


def main():
    print("=" * 75)
    print("TUBITAK RAPORU - Tablo 5: Model Performans Karsilastirmasi (12 Aylik Tahmin)")
    print("=" * 75)

    all_rows = []

    # 1. USD/TRY
    try:
        from src.models.arima_model import (
            ARIMAForecaster, load_data_from_mongodb,
            load_complete_data_from_mongodb, _load_usd_try_from_csv
        )
        usd = load_data_from_mongodb(target_field='usd_try')
        if usd is None or usd.empty:
            usd = load_complete_data_from_mongodb(target_field='usd_try', include_features=False)
        if usd is None or usd.empty:
            usd = _load_usd_try_from_csv()
        if usd is not None and not usd.empty and len(usd) >= 24:
            usd = usd.tail(48) if len(usd) > 48 else usd
            train = usd.iloc[:-STEPS]
            test = usd.iloc[-STEPS:]
            res = run_indicator('USD/TRY', train, test, lambda: ARIMAForecaster(
                target_column='usd_try', seasonal_periods=12, use_log_transform=False, clip_quantiles=(0.02, 0.98)
            ), use_hybrid_forecast=False, svr_C=100.0, svr_epsilon=0.1)
            all_rows.append(('USD/TRY', res))
            print("\nUSD/TRY: ARIMA={}, SVR={}, Hibrit={}".format(res['ARIMA'], res['SVR'], res['Hibrit']))
        else:
            all_rows.append(('USD/TRY', {'ARIMA': None, 'SVR': None, 'Hibrit': None}))
    except Exception as e:
        all_rows.append(('USD/TRY', {'ARIMA': None, 'SVR': None, 'Hibrit': None}))
        print(f"\nUSD/TRY hata: {e}")

    # 2. Enflasyon
    try:
        from src.models.inflation_forecaster import InflationForecaster, load_inflation_data_from_mongodb
        inf = load_inflation_data_from_mongodb(include_features=False)
        if inf is not None and len(inf) >= 24:
            inf = inf.tail(48) if len(inf) > 48 else inf
            train = inf.iloc[:-STEPS]
            test = inf.iloc[-STEPS:]
            res = run_indicator('Enflasyon (TUFE)', train, test, InflationForecaster,
                               use_hybrid_forecast=True, svr_C=1000.0, svr_epsilon=0.02)
            all_rows.append(('Enflasyon (TUFE)', res))
            print("Enflasyon: ARIMA={}, SVR={}, Hibrit={}".format(res['ARIMA'], res['SVR'], res['Hibrit']))
        else:
            all_rows.append(('Enflasyon (TUFE)', {'ARIMA': None, 'SVR': None, 'Hibrit': None}))
    except Exception as e:
        all_rows.append(('Enflasyon (TUFE)', {'ARIMA': None, 'SVR': None, 'Hibrit': None}))
        print(f"Enflasyon hata: {e}")

    # 3. Politika Faizi (TCMB web scraping ile)
    try:
        from src.models.interest_rate_forecaster import InterestRateForecaster, load_interest_rate_data_from_mongodb
        from src.data_collection.tcmb_data_collector import TCMBDataCollector
        
        # Önce MongoDB'den dene
        rate = load_interest_rate_data_from_mongodb(include_features=False)
        
        # MongoDB boşsa TCMB web scraping ile çek
        if rate is None or rate.empty:
            print("  MongoDB'de faiz verisi yok, TCMB web scraping ile çekiliyor...")
            try:
                collector = TCMBDataCollector()
                rate_df = collector.collect_interest_rate_data(years=5)
                if rate_df is not None and not rate_df.empty:
                    rate_df['date'] = pd.to_datetime(rate_df['date'])
                    rate_df = rate_df.drop_duplicates(subset=['date'], keep='last').sort_values('date')
                    rate_df.set_index('date', inplace=True)
                    rate = rate_df['policy_rate'].astype(float)
                    rate = rate.replace([np.inf, -np.inf], np.nan)
                    rate = rate.ffill().bfill()
                    rate = rate.dropna()
                    print(f"  TCMB'den {len(rate)} ay faiz verisi çekildi")
            except Exception as tcmb_err:
                print(f"  TCMB scraping hatası: {tcmb_err}")
        
        # Veriyi temizle ve hazırla
        if rate is not None and not rate.empty:
            rate = rate.astype(float)
            rate = rate.replace([np.inf, -np.inf], np.nan)
            rate = rate.ffill().bfill()
            rate = rate.dropna()
            
            # Aylık frekansa çevir (MS = ay başı)
            if not isinstance(rate.index, pd.DatetimeIndex):
                rate.index = pd.to_datetime(rate.index)
            rate = rate.asfreq('MS', method='ffill')
            rate = rate.dropna()
            
            # Minimum veri kontrolü
            if len(rate) >= 18:
                train = rate.iloc[:-STEPS].copy()
                test = rate.iloc[-STEPS:].copy()
                
                # Son NaN kontrolü
                train = train.dropna()
                test = test.dropna()
                
                if len(train) >= 12 and len(test) >= 6:
                    res = run_indicator('Politika Faizi', train, test, InterestRateForecaster,
                                       use_hybrid_forecast=True, svr_C=100.0, svr_epsilon=0.1, lag=5)
                    all_rows.append(('Politika Faizi', res))
                    print("Politika Faizi: ARIMA={}, SVR={}, Hibrit={}".format(res['ARIMA'], res['SVR'], res['Hibrit']))
                else:
                    all_rows.append(('Politika Faizi', {'ARIMA': None, 'SVR': None, 'Hibrit': None}))
                    print(f"Politika Faizi: Yetersiz veri (train={len(train)}, test={len(test)})")
            else:
                all_rows.append(('Politika Faizi', {'ARIMA': None, 'SVR': None, 'Hibrit': None}))
                print(f"Politika Faizi: Yetersiz veri (toplam={len(rate)})")
        else:
            all_rows.append(('Politika Faizi', {'ARIMA': None, 'SVR': None, 'Hibrit': None}))
            print("Politika Faizi: Veri bulunamadı")
    except Exception as e:
        import traceback
        all_rows.append(('Politika Faizi', {'ARIMA': None, 'SVR': None, 'Hibrit': None}))
        print(f"Politika Faizi hata: {e}")
        print(f"  Traceback: {traceback.format_exc()}")

    # Tablo ciktisi
    print("\n" + "=" * 75)
    print("TABLO 5. MODEL PERFORMANS KARSILASTIRMASI (12 AYLIK TAHMIN)")
    print("=" * 75)
    print(f"{'Gosterge':<20} {'Model':<25} {'RMSE':<10} {'MAPE (%)':<12} {'MAE':<10}")
    print("-" * 75)
    for name, res in all_rows:
        for model in ['ARIMA', 'SVR', 'Hibrit (ARIMA+SVR)']:
            key = 'Hibrit' if 'Hibrit' in model else model
            m = res.get(key)
            if m:
                print(f"{name:<20} {model:<25} {m['RMSE']:<10} {m['MAPE']:<12} {m['MAE']:<10}")
            else:
                print(f"{name:<20} {model:<25} {'—':<10} {'—':<12} {'—':<10}")
    print("=" * 75)
    print("\nBu degerleri TUBITAK raporundaki Tablo 5'e yazabilirsiniz.\n")


if __name__ == '__main__':
    main()
