"""
TUBITAK Raporu - Tablo 6: Capraz Dogrulama Sonuclari (Hibrit Model)
TimeSeriesSplit ile 5 katli capraz dogrulama sonuclarini hesaplar.
Her kat icin Hibrit Model (ARIMA+SVR) MAPE degerlerini verir.
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

def calculate_mape(actual: np.ndarray, pred: np.ndarray) -> float:
    """MAPE (%) hesapla."""
    actual = np.asarray(actual).ravel()
    pred = np.asarray(pred).ravel()
    n = min(len(actual), len(pred))
    actual, pred = actual[:n], pred[:n]
    denom = np.abs(actual) + 1e-10
    mape = np.mean(np.abs((actual - pred) / denom)) * 100
    return round(mape, 2)


def hybrid_cv_mape(data: pd.Series, arima_forecaster_class, n_splits=5,
                   use_hybrid=True, svr_C=100.0, svr_epsilon=0.1, lag=6):
    """
    Hibrit model ile TimeSeriesSplit cross-validation yapar.
    Her kat icin MAPE hesaplar.
    
    Returns:
        dict: {'fold_mape': [fold1, fold2, ...], 'avg_mape': float}
    """
    from src.models.svr_model import SVRForecaster
    
    # Veriyi temizle
    data = data.dropna()
    # 5 katlı CV için minimum veri: her kat için en az 12 train + 3 test = 15, toplam ~60-70 veri
    if len(data) < 50:  # Minimum veri gereksinimi (5 kat için)
        return {'fold_mape': [], 'avg_mape': None}
    
    # TimeSeriesSplit oluştur
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_mape_list = []
    fold_details = []
    
    fold_num = 0
    for train_idx, test_idx in tscv.split(data):
        fold_num += 1
        
        train_fold = data.iloc[train_idx]
        test_fold = data.iloc[test_idx]
        
        # Minimum train ve test boyutları
        if len(train_fold) < 10 or len(test_fold) < 2:
            print(f"  Kat {fold_num}: Yetersiz veri (train={len(train_fold)}, test={len(test_fold)}), atlaniyor.")
            continue
        
        try:
            # ARIMA modeli eğit
            arima_fc = arima_forecaster_class()
            arima_fc.fit(train_fold, test_size=0.0)
            
            # ARIMA tahmini
            steps = len(test_fold)
            try:
                arima_res = arima_fc.forecast(steps=steps, use_hybrid=False)
            except TypeError:
                arima_res = arima_fc.forecast(steps=steps)
            
            if not arima_res or 'forecast' not in arima_res:
                print(f"  Kat {fold_num}: ARIMA tahmin uretilemedi.")
                continue
            
            arima_pred = np.asarray(arima_res['forecast'].values)[:steps]
            actual = test_fold.values[:steps]
            
            # Hibrit model için SVR tahmini
            hybrid_pred = None
            
            if use_hybrid:
                try:
                    # SVR modeli eğit
                    lag_use = min(lag, max(3, len(train_fold) // 2 - 1))
                    svr = SVRForecaster(kernel='rbf', C=svr_C, epsilon=svr_epsilon)
                    svr.fit(train_fold, lag=lag_use)
                    svr_pred_series = svr.predict(train_fold, steps=steps)
                    svr_pred = np.asarray(svr_pred_series.values)[:steps]
                    
                    # Hibrit tahmin (ARIMA %60 + SVR %40)
                    n = min(len(arima_pred), len(svr_pred), len(actual))
                    hybrid_pred = 0.6 * arima_pred[:n] + 0.4 * svr_pred[:n]
                    actual_hybrid = actual[:n]
                except Exception as svr_err:
                    print(f"  Kat {fold_num}: SVR hatasi: {svr_err}, sadece ARIMA kullaniliyor.")
                    hybrid_pred = arima_pred[:len(actual)]
                    actual_hybrid = actual
            else:
                hybrid_pred = arima_pred[:len(actual)]
                actual_hybrid = actual
            
            # MAPE hesapla
            mape_fold = calculate_mape(actual_hybrid, hybrid_pred)
            fold_mape_list.append(mape_fold)
            fold_details.append({
                'fold': fold_num,
                'mape': mape_fold,
                'train_size': len(train_fold),
                'test_size': len(test_fold)
            })
            
            print(f"  Kat {fold_num}: MAPE = {mape_fold:.2f}%")
            
        except Exception as e:
            print(f"  Kat {fold_num}: Hata: {e}")
            # Hata durumunda da devam et, ama bu katı atla
            continue
    
    if len(fold_mape_list) == 0:
        return {'fold_mape': [], 'avg_mape': None}
    
    avg_mape = round(np.mean(fold_mape_list), 2)
    return {'fold_mape': fold_mape_list, 'avg_mape': avg_mape, 'details': fold_details}


def main():
    print("=" * 80)
    print("TUBITAK RAPORU - Tablo 6: Capraz Dogrulama Sonuclari (Hibrit Model)")
    print("=" * 80)
    print("TimeSeriesSplit ile 5 katli capraz dogrulama (Hibrit Model: ARIMA+SVR)\n")
    
    all_results = []
    
    # 1. USD/TRY
    print("\n[1/3] USD/TRY analizi...")
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
            # 5 katlı CV için daha fazla veri gerekli (minimum 60 ay)
            # Son 60 ayı kullan (5 kat için yeterli veri sağlar)
            usd = usd.tail(60) if len(usd) > 60 else usd
            
            res = hybrid_cv_mape(
                usd,
                lambda: ARIMAForecaster(
                    target_column='usd_try',
                    seasonal_periods=12,
                    use_log_transform=False,
                    clip_quantiles=(0.02, 0.98)
                ),
                n_splits=5,
                use_hybrid=True,
                svr_C=100.0,
                svr_epsilon=0.1,
                lag=6
            )
            
            if res['fold_mape']:
                all_results.append(('USD/TRY', res))
                print(f"  [OK] Ortalama MAPE: {res['avg_mape']:.2f}%")
            else:
                all_results.append(('USD/TRY', {'fold_mape': [], 'avg_mape': None}))
                print("  [HATA] Sonuc uretilemedi")
        else:
            all_results.append(('USD/TRY', {'fold_mape': [], 'avg_mape': None}))
            print("  [HATA] Yetersiz veri")
    except Exception as e:
        import traceback
        all_results.append(('USD/TRY', {'fold_mape': [], 'avg_mape': None}))
        print(f"  [HATA] {e}")
        print(f"  Traceback: {traceback.format_exc()}")
    
    # 2. Enflasyon
    print("\n[2/3] Enflasyon analizi...")
    try:
        from src.models.inflation_forecaster import InflationForecaster, load_inflation_data_from_mongodb
        inf = load_inflation_data_from_mongodb(include_features=False)
        
        if inf is not None and not inf.empty and len(inf) >= 24:
            # 5 katlı CV için daha fazla veri gerekli (minimum 60 ay)
            # Son 60 ayı kullan
            inf = inf.tail(60) if len(inf) > 60 else inf
            
            res = hybrid_cv_mape(
                inf,
                InflationForecaster,
                n_splits=5,
                use_hybrid=True,
                svr_C=1000.0,
                svr_epsilon=0.02,
                lag=6
            )
            
            if res['fold_mape']:
                all_results.append(('Enflasyon', res))
                print(f"  [OK] Ortalama MAPE: {res['avg_mape']:.2f}%")
            else:
                all_results.append(('Enflasyon', {'fold_mape': [], 'avg_mape': None}))
                print("  [HATA] Sonuc uretilemedi")
        else:
            all_results.append(('Enflasyon', {'fold_mape': [], 'avg_mape': None}))
            print("  [HATA] Yetersiz veri")
    except Exception as e:
        import traceback
        all_results.append(('Enflasyon', {'fold_mape': [], 'avg_mape': None}))
        print(f"  [HATA] {e}")
        print(f"  Traceback: {traceback.format_exc()}")
    
    # 3. Politika Faizi
    print("\n[3/3] Politika Faizi analizi...")
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
            
            # Aylık frekansa çevir
            if not isinstance(rate.index, pd.DatetimeIndex):
                rate.index = pd.to_datetime(rate.index)
            rate = rate.asfreq('MS', method='ffill')
            rate = rate.dropna()
            
            # 5 katlı CV için minimum 36 ay veri gerekli
            if len(rate) >= 36:
                res = hybrid_cv_mape(
                    rate,
                    InterestRateForecaster,
                    n_splits=5,
                    use_hybrid=True,
                    svr_C=100.0,
                    svr_epsilon=0.1,
                    lag=5
                )
                
                if res['fold_mape']:
                    all_results.append(('Politika Faizi', res))
                    print(f"  [OK] Ortalama MAPE: {res['avg_mape']:.2f}%")
                else:
                    all_results.append(('Politika Faizi', {'fold_mape': [], 'avg_mape': None}))
                    print("  [HATA] Sonuc uretilemedi")
            else:
                all_results.append(('Politika Faizi', {'fold_mape': [], 'avg_mape': None}))
                print(f"  [HATA] Yetersiz veri (toplam={len(rate)})")
        else:
            all_results.append(('Politika Faizi', {'fold_mape': [], 'avg_mape': None}))
            print("  [HATA] Veri bulunamadi")
    except Exception as e:
        import traceback
        all_results.append(('Politika Faizi', {'fold_mape': [], 'avg_mape': None}))
        print(f"  [HATA] {e}")
        print(f"  Traceback: {traceback.format_exc()}")
    
    # Tablo çıktısı
    print("\n" + "=" * 80)
    print("TABLO 6. CAPRAZ DOGRULAMA SONUCLARI (HIBRIT MODEL)")
    print("=" * 80)
    print(f"{'Gösterge':<20} {'Kat 1 MAPE (%)':<18} {'Kat 2 MAPE (%)':<18} {'Kat 3 MAPE (%)':<18} {'Kat 4 MAPE (%)':<18} {'Kat 5 MAPE (%)':<18} {'Ortalama MAPE':<15}")
    print("-" * 80)
    
    for name, res in all_results:
        fold_mape = res.get('fold_mape', [])
        avg_mape = res.get('avg_mape')
        
        if fold_mape and len(fold_mape) >= 5:
            # 5 kat için MAPE değerleri
            kat1 = fold_mape[0] if len(fold_mape) > 0 else '—'
            kat2 = fold_mape[1] if len(fold_mape) > 1 else '—'
            kat3 = fold_mape[2] if len(fold_mape) > 2 else '—'
            kat4 = fold_mape[3] if len(fold_mape) > 3 else '—'
            kat5 = fold_mape[4] if len(fold_mape) > 4 else '—'
            avg = avg_mape if avg_mape is not None else '—'
            
            print(f"{name:<20} {kat1:<18} {kat2:<18} {kat3:<18} {kat4:<18} {kat5:<18} {avg:<15}")
        elif fold_mape:
            # Eksik katlar için '—' göster
            kat_values = fold_mape + ['—'] * (5 - len(fold_mape))
            avg = avg_mape if avg_mape is not None else '—'
            print(f"{name:<20} {kat_values[0]:<18} {kat_values[1]:<18} {kat_values[2]:<18} {kat_values[3]:<18} {kat_values[4]:<18} {avg:<15}")
        else:
            print(f"{name:<20} {'—':<18} {'—':<18} {'—':<18} {'—':<18} {'—':<18} {'—':<15}")
    
    print("=" * 80)
    print("\nBu değerleri TUBITAK raporundaki Tablo 6'ya yazabilirsiniz.\n")
    
    # Detaylı sonuçlar
    print("\n" + "=" * 80)
    print("DETAYLI SONUCLAR:")
    print("=" * 80)
    for name, res in all_results:
        if res.get('details'):
            print(f"\n{name}:")
            for detail in res['details']:
                print(f"  Kat {detail['fold']}: MAPE = {detail['mape']:.2f}% (train={detail['train_size']}, test={detail['test_size']})")
            if res.get('avg_mape'):
                print(f"  Ortalama MAPE: {res['avg_mape']:.2f}%")


if __name__ == '__main__':
    main()
