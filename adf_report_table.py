"""
TÜBİTAK Raporu - Tablo 3: ADF Durağanlık Testi Sonuçları
Projedeki gerçek verilerle ADF testi çalıştırır ve doğru değerleri üretir.
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller


def adf_pvalue(series):
    """ADF testi uygula, p-değerini döndür."""
    clean = series.dropna()
    if len(clean) < 8:
        return None
    try:
        result = adfuller(clean, autolag='AIC')
        return round(result[1], 4)  # p-value
    except Exception:
        return None


def get_d_order(series, max_d=2):
    """p < 0.05 olana kadar fark al; d parametresini döndür."""
    current = series.dropna()
    for d in range(max_d + 1):
        p = adf_pvalue(current)
        if p is not None and p < 0.05:
            return d
        current = current.diff().dropna()
    return max_d


def main():
    print("=" * 70)
    print("TÜBİTAK RAPORU - Tablo 3: ADF Durağanlık Testi Sonuçları")
    print("(Proje verileriyle hesaplanan gerçek değerler)")
    print("=" * 70)

    results = []

    # 1. USD/TRY
    try:
        from src.models.arima_model import load_data_from_mongodb, load_complete_data_from_mongodb, _load_usd_try_from_csv
        usd = load_data_from_mongodb(target_field='usd_try')
        if usd is None or usd.empty:
            usd = load_complete_data_from_mongodb(target_field='usd_try', include_features=False)
        if usd is None or usd.empty:
            usd = _load_usd_try_from_csv()
        if usd is not None and not usd.empty:
            # Rapor penceresi: son 48 ay (uygulama ile uyumlu)
            usd_48 = usd.tail(48) if len(usd) > 48 else usd
            p_raw = adf_pvalue(usd_48)
            d = get_d_order(usd_48)
            # "Fark alınmış p" = d. fark sonrası p (durağanlık sağlayan)
            s = usd_48
            for _ in range(d):
                s = s.diff().dropna()
            p_diff = adf_pvalue(s) if len(s) >= 8 else None
            results.append(('USD/TRY', p_raw, p_diff, d))
            print(f"\nUSD/TRY: n={len(usd_48)}, Ham p={p_raw}, Fark(d={d}) p={p_diff}, d={d}")
        else:
            results.append(('USD/TRY', None, None, None))
            print("\nUSD/TRY: Veri yok")
    except Exception as e:
        results.append(('USD/TRY', None, None, None))
        print(f"\nUSD/TRY: Hata - {e}")

    # 2. Enflasyon
    try:
        from src.models.inflation_forecaster import load_inflation_data_from_mongodb
        inf = load_inflation_data_from_mongodb(include_features=False)
        if inf is not None and not inf.empty:
            p_raw = adf_pvalue(inf)
            d = get_d_order(inf)
            s = inf
            for _ in range(d):
                s = s.diff().dropna()
            p_diff = adf_pvalue(s) if len(s) >= 8 else None
            results.append(('Enflasyon (TÜFE)', p_raw, p_diff, d))
            print(f"Enflasyon: n={len(inf)}, Ham p={p_raw}, Fark(d={d}) p={p_diff}, d={d}")
        else:
            results.append(('Enflasyon (TÜFE)', None, None, None))
            print("Enflasyon: Veri yok")
    except Exception as e:
        results.append(('Enflasyon (TÜFE)', None, None, None))
        print(f"Enflasyon: Hata - {e}")

    # 3. Politika Faizi
    try:
        from src.models.interest_rate_forecaster import load_interest_rate_data_from_mongodb
        rate = load_interest_rate_data_from_mongodb(include_features=False)
        if rate is not None and not rate.empty:
            p_raw = adf_pvalue(rate)
            d = get_d_order(rate)
            s = rate
            for _ in range(d):
                s = s.diff().dropna()
            p_diff = adf_pvalue(s) if len(s) >= 8 else None
            results.append(('Politika Faizi', p_raw, p_diff, d))
            print(f"Politika Faizi: n={len(rate)}, Ham p={p_raw}, Fark(d={d}) p={p_diff}, d={d}")
        else:
            results.append(('Politika Faizi', None, None, None))
            print("Politika Faizi: Veri yok")
    except Exception as e:
        results.append(('Politika Faizi', None, None, None))
        print(f"Politika Faizi: Hata - {e}")

    # Tablo çıktısı
    print("\n" + "=" * 70)
    print("TABLO 3. ADF DURAĞANLIK TESTİ SONUÇLARI (Raporda kullanılacak değerler)")
    print("=" * 70)
    print(f"{'Gösterge':<25} {'Ham Veri p-değeri':<20} {'Fark Alınmış p-değeri':<22} {'d parametresi':<12}")
    print("-" * 70)
    for name, p_raw, p_diff, d in results:
        pr = f"{p_raw:.3f}" if p_raw is not None else "—"
        pd_ = f"{p_diff:.3f}" if p_diff is not None else "—"
        dd = str(d) if d is not None else "—"
        print(f"{name:<25} {pr:<20} {pd_:<22} {dd:<12}")
    print("=" * 70)
    print("\nYorum: Ham veri p > 0.05 ise seri durağan değil; fark alınmış p < 0.05 ise d=1 ile durağan.")
    print("Bu değerleri TÜBİTAK raporundaki Tablo 3'e aynen yazabilirsiniz.\n")


if __name__ == '__main__':
    main()
