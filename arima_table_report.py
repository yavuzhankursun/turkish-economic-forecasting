"""
TÜBİTAK Raporu - Tablo 4: Optimal ARIMA Parametreleri
Projedeki gerçek verilerle ARIMA eğitir ve (p,d,q) + AIC değerlerini üretir.
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import pandas as pd

def get_aic(forecaster):
    """Eğitilmiş modelden AIC al (pmdarima)."""
    if forecaster.fitted_model is None:
        return None
    try:
        return round(float(forecaster.fitted_model.aic()), 2)
    except Exception:
        return None

def main():
    print("=" * 70)
    print("TÜBİTAK RAPORU - Tablo 4: Optimal ARIMA Parametreleri")
    print("(auto_arima ile proje verilerinden hesaplanan gerçek değerler)")
    print("=" * 70)

    results = []

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
        if usd is not None and not usd.empty:
            usd_48 = usd.tail(48) if len(usd) > 48 else usd
            forecaster = ARIMAForecaster(
                target_column='usd_try',
                seasonal_periods=12,
                use_log_transform=False,
                clip_quantiles=(0.02, 0.98)
            )
            forecaster.fit(usd_48, test_size=0.2)
            order = forecaster.best_params
            aic = get_aic(forecaster)
            results.append(('USD/TRY', order, aic))
            print(f"\nUSD/TRY: (p,d,q)={order}, AIC={aic}")
        else:
            results.append(('USD/TRY', None, None))
            print("\nUSD/TRY: Veri yok")
    except Exception as e:
        results.append(('USD/TRY', None, None))
        print(f"\nUSD/TRY: Hata - {e}")

    # 2. Enflasyon
    try:
        from src.models.inflation_forecaster import InflationForecaster, load_inflation_data_from_mongodb
        inf = load_inflation_data_from_mongodb(include_features=False)
        if inf is not None and not inf.empty:
            inf_48 = inf.tail(48) if len(inf) > 48 else inf
            forecaster = InflationForecaster()
            forecaster.fit(inf_48, test_size=0.2)
            order = forecaster.best_params
            aic = get_aic(forecaster)
            results.append(('Enflasyon (TÜFE)', order, aic))
            print(f"Enflasyon: (p,d,q)={order}, AIC={aic}")
        else:
            results.append(('Enflasyon (TÜFE)', None, None))
            print("Enflasyon: Veri yok")
    except Exception as e:
        results.append(('Enflasyon (TÜFE)', None, None))
        print(f"Enflasyon: Hata - {e}")

    # 3. Politika Faizi
    try:
        from src.models.interest_rate_forecaster import InterestRateForecaster, load_interest_rate_data_from_mongodb
        rate = load_interest_rate_data_from_mongodb(include_features=False)
        if rate is not None and not rate.empty:
            # Rapor için yeterli veri kullan (faiz için 12 ay çok az; 24+ kullan)
            rate_use = rate.tail(36) if len(rate) > 36 else rate
            forecaster = InterestRateForecaster()
            forecaster.fit(rate_use, test_size=0.2)
            order = forecaster.best_params
            aic = get_aic(forecaster)
            results.append(('Politika Faizi', order, aic))
            print(f"Politika Faizi: (p,d,q)={order}, AIC={aic}")
        else:
            results.append(('Politika Faizi', None, None))
            print("Politika Faizi: Veri yok")
    except Exception as e:
        results.append(('Politika Faizi', None, None))
        print(f"Politika Faizi: Hata - {e}")

    # Tablo çıktısı
    print("\n" + "=" * 70)
    print("TABLO 4. OPTIMAL ARIMA PARAMETRELERI (Raporda kullanilacak degerler)")
    print("=" * 70)
    print(f"{'Gösterge':<22} {'(p,d,q) Parametreleri':<28} {'AIC Değeri':<12}")
    print("-" * 70)
    for name, order, aic in results:
        ord_str = f"({order[0]}, {order[1]}, {order[2]})" if order else "—"
        aic_str = str(aic) if aic is not None else "—"
        print(f"{name:<22} {ord_str:<28} {aic_str:<12}")
    print("=" * 70)
    print("\nYorum: auto_arima algoritmasi ile AIC kriterine gore secilen parametreler.")
    print("Bu degerleri TUBITAK raporundaki Tablo 4'e aynen yazabilirsiniz.\n")


if __name__ == '__main__':
    main()
