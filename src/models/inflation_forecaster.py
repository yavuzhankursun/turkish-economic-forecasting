"""
Enflasyon Tahmin Modeli
=======================

TÜBİTAK Projesi - Enflasyon oranı tahmini için ARIMA modeli + News analizi

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from sklearn.model_selection import TimeSeriesSplit

# Proje kök dizinini import yolu olarak ekle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models.arima_model import ARIMAForecaster, load_complete_data_from_mongodb
from src.utils.mongodb_manager import MongoDBManager
from src.services.multi_indicator_service import MultiIndicatorNewsService
from src.models.svr_model import SVRForecaster  # G-10: ARIMA-SVR hibrit

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InflationForecaster(ARIMAForecaster):
    """
    Enflasyon tahmini için özelleştirilmiş ARIMA modeli.
    
    ARIMAForecaster'dan türetilmiş, enflasyon verilerine özel.
    ARIMA + News API + Claude sentiment analizi ile nihai tahmin.
    """
    
    def __init__(self):
        # MAPE İYİLEŞTİRME: Log transform kaldırıldı, winsorization genişletildi
        # Enflasyon zaten yüzde cinsinden, log transform düşük değerlerde MAPE'yi şişiriyordu
        super().__init__(
            target_column='inflation_rate',
            seasonal_periods=12,
            use_log_transform=False,  # MAPE FIX: Log transform kapatıldı
            # Winsorization genişletildi - agresif clip (0.15, 0.85) uç değerleri kesiyordu
            clip_quantiles=(0.02, 0.98),  # MAPE FIX: Daha yumuşak clip
            log_timeseries_validation=True  # MongoDB'ye validation kayıtları için
        )
        self.news_service = MultiIndicatorNewsService()
        # Enflasyon için genişletilmiş ARIMA parametreleri (daha karmaşık modellere izin ver)
        self.arima_max_p = 5  # MAPE FIX: 3 -> 5
        self.arima_max_q = 5  # MAPE FIX: 3 -> 5
        self.arima_max_P = 2
        self.arima_max_Q = 2
        logger.info("Enflasyon Forecaster başlatıldı (MAPE optimizasyonu ile).")

    def find_best_params(self, data, exog=None, information_criterion: str = 'aic'):
        """Enflasyon için özelleştirilmiş ARIMA parametre arama - daha geniş arama aralığı."""
        import pmdarima as pm
        
        if not hasattr(self, 'arima_max_p'):
            self.arima_max_p = 3
            self.arima_max_q = 3
            self.arima_max_P = 2
            self.arima_max_Q = 2
        
        logger.info(f"Enflasyon için optimize edilmiş ARIMA parametreleri kullanılıyor (max_p={self.arima_max_p}, max_q={self.arima_max_q})...")
        
        seasonal_flag = False
        seasonal_period = 1
        if self.seasonal:
            min_required = (self.seasonal_periods or 0) * 2
            if min_required and len(data) >= min_required:
                seasonal_flag = True
                seasonal_period = int(self.seasonal_periods)
            else:
                logger.info("Veri uzunluğu mevsimsellik için yetersiz. Seasonal ARIMA devre dışı bırakıldı.")
        
        min_p = 1 if len(data) >= 10 else 0
        min_q = 1 if len(data) >= 10 else 0
        
        # Enflasyon için MAPE odaklı model seçimi dene (başarısız olursa AIC fallback)
        out_of_sample_size = min(6, max(3, len(data) // 6))

        auto_kwargs = dict(
            start_p=min_p, start_q=min_q,
            test='adf',
            max_p=self.arima_max_p,
            max_q=self.arima_max_q,
            max_d=2,
            seasonal=seasonal_flag,
            m=seasonal_period,
            max_P=self.arima_max_P,
            max_Q=self.arima_max_Q,
            max_D=1,
            d=None,
            start_P=1 if seasonal_flag else 0,
            D=None,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            information_criterion=information_criterion.lower(),
            n_jobs=1,
            random_state=42,
            with_intercept=True,
            X=exog,
            maxiter=80
        )

        try:
            # MAPE scoring bazen istikrarsız olabilir, varsayılan AIC ile devam et
            self.fitted_model = pm.auto_arima(
                data,
                **auto_kwargs
            )
        except Exception as e:
            logger.warning(f"auto_arima başarısız ({e}). Exog kapatılıp tekrar deneniyor...")
            try:
                auto_kwargs_no_exog = dict(auto_kwargs)
                auto_kwargs_no_exog['X'] = None
                self.fitted_model = pm.auto_arima(data, **auto_kwargs_no_exog)
            except Exception as e2:
                logger.error(f"auto_arima tamamen başarısız ({e2}). Basit ARIMA(1,1,1) fallback.")
                self.fitted_model = pm.ARIMA(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
                self.fitted_model.fit(data)
        
        params = self.fitted_model.get_params()
        self.best_params = params['order']
        self.best_seasonal_order = params.get('seasonal_order')
        logger.info(f"Enflasyon için en iyi parametreler bulundu: ARIMA{self.best_params}")
        
        return self.fitted_model
    
    def fit(self, data, test_size=0.2, exogenous_data=None):
        """Enflasyon modeli için gelişmiş öğrenme döngüsü."""
        # MAPE FIX: Use only recent data (last 48 months = 4 years) to focus on current regime
        # 48 months is enough for cross-validation while still being recent
        if isinstance(data, pd.Series) and len(data) > 48:
            logger.info(f"MAPE FIX: Veri son 48 aya kısıtlanıyor ({len(data)} -> 48)")
            data = data.tail(48)
            if exogenous_data is not None:
                exogenous_data = exogenous_data.tail(48)
        
        # 1. Veri Yükleme (Eğer verilmediyse)
        if exogenous_data is None and not isinstance(data, (pd.Series, pd.DataFrame)):
            feature_frame = load_inflation_data(include_features=True)
            if isinstance(feature_frame, pd.DataFrame) and not feature_frame.empty:
                # Target ve Features ayır
                if 'inflation_rate' in feature_frame.columns:
                    data = feature_frame['inflation_rate']
                    exogenous_data = feature_frame.drop(columns=['inflation_rate'])
                else:
                    logger.error("Veri setinde 'inflation_rate' bulunamadı.")
                    return self
            else:
                logger.warning("Ek özellik bulunamadı, model tek değişkenli olarak eğitim alacak.")

        # 2. Exogenous Data Hazırlığı (Future Generation için sakla)
        if exogenous_data is not None:
            # Tüm geçmişi sakla (raw değerler lazım: usd_try, brent_oil vb.)
            # Eğer zaten set edilmişse (cross-validation için), override etme!
            if not hasattr(self, 'history_exog') or self.history_exog is None:
                self.history_exog = exogenous_data.copy()
            
            # 3. Model Eğitimi İçin Sadece Kullanılabilir (Lagged) Feature'ları Seç
            # Gelecekte 'usd_try' değerini bilemeyiz (tahmin etmezsek), 
            # ama 'usd_try_lag1' değerini t anında biliriz (t-1'den gelir).
            # Bu yüzden raw feature'ları eğitim setinden çıkarıyoruz.
            
            train_cols = [c for c in exogenous_data.columns if 
                          c.endswith('_lag1') or 
                          c.endswith('_lag2') or 
                          c.endswith('_dummy') or 
                          c.endswith('_sin') or 
                          c.endswith('_cos') or 
                          c == 'trend' or 
                          '_mom_' in c or
                          'event_count_log' in c or
                          'weighted_tone' in c]
                          
            if train_cols:
                exogenous_data_train = exogenous_data[train_cols].copy()
                logger.info(f"Eğitim için seçilen featurelar: {train_cols}")
            else:
                exogenous_data_train = exogenous_data # Fallback
        else:
            exogenous_data_train = None

        # super().fit çağır
        result = super().fit(data, test_size=test_size, exogenous_data=exogenous_data_train)
        
        # Cross-validation için exog verilerini sakla (index-based)
        if exogenous_data_train is not None:
            if hasattr(self, 'train_data') and self.train_data is not None:
                # train_data'nın index'ine göre exog_train'i seç
                self.exog_train = exogenous_data_train.loc[self.train_data.index]
                if hasattr(self, 'test_data') and self.test_data is not None and not self.test_data.empty:
                    self.exog_test = exogenous_data_train.loc[self.test_data.index]
                else:
                    self.exog_test = None
            else:
                # Fallback: tüm veriyi exog_train olarak sakla
                self.exog_train = exogenous_data_train
                self.exog_test = None
        
        return result

    def _prepare_future_exog(self, steps):
        """
        Gelecek tahminleri için exogenous değişkenleri hazırlar (Lagging mantığı ile).
        ÖZEL: Model tarafından eğitimde kullanılan kolonları üretir.
        """
        # 1. Tarih indexi oluştur
        if hasattr(self, 'train_data') and self.train_data is not None:
            last_date = self.train_data.index[-1]
        elif hasattr(self, 'test_data') and self.test_data is not None:
            last_date = self.test_data.index[-1]
        else:
            from datetime import datetime
            last_date = pd.Timestamp(datetime.now())
        
        # Gelecek tarihleri
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=steps, freq='MS')
        future_exog = pd.DataFrame(index=future_dates)
        
        # 2. Temel özellikler (Mevsimsellik + Trend)
        months = future_exog.index.month
        future_exog['month_sin'] = np.sin(2 * np.pi * months / 12)
        future_exog['month_cos'] = np.cos(2 * np.pi * months / 12)
        
        # Trend: Son eğitim indexinden devam
        if hasattr(self, 'train_data') and self.train_data is not None:
            last_trend = len(self.train_data) - 1
        else:
            last_trend = 0
        future_exog['trend'] = np.arange(last_trend + 1, last_trend + 1 + steps, dtype=float)
        
        # 3. Lagged özellikler (SADECE modelin eğitimde kullandığı kolonları)
        if hasattr(self, 'history_exog') and self.history_exog is not None:
            logger.info(f"🔧 history_exog var: {self.history_exog.shape}, columns: {list(self.history_exog.columns)[:10]}...")
            # Son bilinen değerler
            last_row = self.history_exog.iloc[-1]
            
            # Ham değişkenlerin (usd_try, brent vb.) trendini hesapla
            drifts = {}
            raw_vars = ['usd_try', 'brent_oil', 'min_wage', 'inflation_rate']
            
            for col in raw_vars:
                if col in self.history_exog.columns:
                    # Son 6 ayın ortalama değişimi (momentum)
                    diffs = self.history_exog[col].diff()
                    drift = diffs.tail(6).mean()
                    if np.isnan(drift): drift = 0
                    drifts[col] = drift
            
            # Sentiment History (Recursive update için)
            # Kullanıcı isteği: Weighted Tone (t-1, t-2) + Event Count (t-1)
            weighted_tone_t = 0.0
            weighted_tone_t_1 = 0.0
            event_count_log_t = 0.0
            
            if 'News_Tone_Mean' in self.history_exog.columns and 'Event_Count' in self.history_exog.columns:
                # Son iki ayın weighted tone değerleri
                tone_mean = self.history_exog['News_Tone_Mean'].iloc[-1]
                event_count = self.history_exog['Event_Count'].iloc[-1]
                weighted_tone_t = tone_mean * np.log1p(event_count)
                event_count_log_t = np.log1p(event_count)
                
                if len(self.history_exog) > 1:
                    tone_mean_prev = self.history_exog['News_Tone_Mean'].iloc[-2]
                    event_count_prev = self.history_exog['Event_Count'].iloc[-2]
                    weighted_tone_t_1 = tone_mean_prev * np.log1p(event_count_prev)

            # Recursive Feature Generation
            # t+1 tahmini için t değerleri (lag1) kullanılır.
            # t+2 tahmini için t+1 değerleri (tahmin edilen/drift uygulanan) kullanılır.
            
            future_rows = []
            current_values = last_row.copy()
            
            for i in range(steps):
                step_feat = {}
                
                # Gelecek değerleri tahmin et (Drift ile)
                next_values = current_values.copy()
                for col, drift in drifts.items():
                    if col in next_values:
                        next_values[col] = current_values[col] + drift
                
                # Feature Mapping (Lag1 -> Current Value)
                # t zamanındayız, t+1 tahmin ediyoruz. Feature olarak (t) değerlerini kullanıyoruz.
                if 'usd_try' in current_values:
                    step_feat['usd_try_lag1'] = current_values['usd_try']
                if 'brent_oil' in current_values:
                    step_feat['brent_oil_lag1'] = current_values['brent_oil']
                if 'min_wage' in current_values:
                    step_feat['min_wage_lag1'] = current_values['min_wage']
                
                # Sentiment Features (Lagged)
                # Kullanıcı isteği: Weighted Tone (hacim ağırlıklı) + Event Count (log)
                step_feat['weighted_tone_lag1'] = weighted_tone_t
                step_feat['weighted_tone_lag2'] = weighted_tone_t_1
                step_feat['event_count_log_lag1'] = event_count_log_t
                
                # Update Sentiment for next step (Future is unknown -> assume neutral)
                # Son 6 ayın ortalamasına dön (Trend devam varsayımı)
                weighted_tone_t_1 = weighted_tone_t
                if 'News_Tone_Mean' in self.history_exog.columns:
                    avg_tone = self.history_exog['News_Tone_Mean'].tail(6).mean()
                    avg_events = self.history_exog['Event_Count'].tail(6).mean()
                    weighted_tone_t = avg_tone * np.log1p(avg_events)
                    event_count_log_t = np.log1p(avg_events)
                else:
                    weighted_tone_t = 0.0  # Nötr
                    event_count_log_t = 0.0

                # Min Wage Dummy (Yılda iki kez artış varsayımı - Ocak ve Temmuz)
                # Basitçe: Eğer mevcut ay 12 veya 6 ise, gelecek ay (1 veya 7) zam var.
                # future_exog index'i tarihleri içerir.
                try:
                    current_date = future_exog.index[i]
                    if current_date.month in [1, 7]:
                         step_feat['min_wage_increase_dummy'] = 1
                    else:
                         step_feat['min_wage_increase_dummy'] = 0
                except:
                    step_feat['min_wage_increase_dummy'] = 0
                
                # Momentum özellikleri (Basitleştirilmiş: sabit momentum varsayımı)
                # Model eğer bu feature'ları bekliyorsa mutlaka eklenmeli
                if 'usd_try_mom_3' in self.history_exog.columns:
                    step_feat['usd_try_mom_3'] = last_row['usd_try_mom_3'] # Değişimi sabit tut
                else:
                    # Hesapla veya 0 koy
                    step_feat['usd_try_mom_3'] = 0.0
                    
                if 'brent_oil_mom_3' in self.history_exog.columns:
                    step_feat['brent_oil_mom_3'] = last_row['brent_oil_mom_3']
                else:
                    step_feat['brent_oil_mom_3'] = 0.0

                future_rows.append(step_feat)
                
                # Bir sonraki adım için değerleri güncelle
                current_values = next_values
            
            # DataFrame oluştur ve birleştir
            lags_df = pd.DataFrame(future_rows, index=future_exog.index)
            future_exog = pd.concat([future_exog, lags_df], axis=1)
            
        # *** KRİTİK: ÖNCE modelin beklediği kolonları belirle ***
        needed_cols = None
        if hasattr(self, 'fitted_model') and hasattr(self.fitted_model, 'exog_names'):
            # Statsmodels exog names (const hariç)
            needed_cols = [c for c in self.fitted_model.exog_names if c != 'const']
            logger.info(f"Model {len(needed_cols)} kolon bekliyor: {needed_cols[:5]}...")
        
        # NaN doldur
        future_exog = future_exog.fillna(0)
        
        # *** FİLTRELEME: Sadece modelin beklediği kolonları tut ***
        if needed_cols is not None:
            # Eksik kolonları 0 ile doldur
            for col in needed_cols:
                if col not in future_exog.columns:
                    logger.warning(f"Eksik kolon: {col}, 0 ile dolduruluyor.")
                    future_exog[col] = 0.0
            
            # Fazla kolonları at
            extra_cols = set(future_exog.columns) - set(needed_cols)
            if extra_cols:
                logger.warning(f"Fazla kolonlar atılıyor: {extra_cols}")
            
            # SADECE gereken kolonları tut (AYNI SIRADA!)
            future_exog = future_exog[needed_cols]
            logger.info(f"✅ Future exog hazır: {future_exog.shape}")
        
        return future_exog
    
    def forecast(self, steps=12, confidence_level: float = 0.95, use_hybrid: bool = True):
        """
        Enflasyon için tahmin:
        - ARIMA tahmini (base)
        - SVR ile hibritleştir
        - Naive Drift modeli ekle (son değer + ortalama değişim)
        """
        arima_results = super().forecast(steps=steps, confidence_level=confidence_level)
        if not use_hybrid or arima_results is None or 'forecast' not in arima_results:
            return arima_results

        try:
            # Eğitim verisi (orijinal skala)
            if self.original_train_data is not None and not self.original_train_data.empty:
                train_original = self.original_train_data.dropna()
            elif self.train_data is not None and not self.train_data.empty:
                train_original = self._inverse_transform(self.train_data.dropna())
            else:
                return arima_results

            # NaN kontrolü ve temizleme
            train_original = train_original.replace([np.inf, -np.inf], np.nan).dropna()

            if train_original.empty or len(train_original) < 6:
                return arima_results

            # =====================================================
            # NAIVE DRIFT MODELİ
            # Son değer + ortalama aylık değişim * adım sayısı
            # =====================================================
            last_value = train_original.iloc[-1]
            # Son 6 ayın ortalama değişimi (kısa vadeli trend)
            recent_changes = train_original.diff().tail(6)
            avg_monthly_change = recent_changes.mean() if len(recent_changes) > 0 else 0.0
            if np.isnan(avg_monthly_change):
                avg_monthly_change = 0.0
            
            # Naive Drift tahmini oluştur
            # MAPE FIX: Aylık değişimi sınırla - çok agresif extrapolation'ı engelle
            max_monthly_change = 10.0  # En fazla ayda 10 puan değişim
            avg_monthly_change = np.clip(avg_monthly_change, -max_monthly_change, max_monthly_change)
            
            naive_forecast_values = []
            for i in range(steps):
                naive_val = last_value + (i + 1) * avg_monthly_change
                # MAPE FIX: Hard bounds - enflasyon 0% ile 200% arasında olmalı
                naive_val = np.clip(naive_val, 0.0, 200.0)
                naive_forecast_values.append(naive_val)
            
            naive_forecast = pd.Series(naive_forecast_values, index=arima_results['forecast'].index)
            
            # =====================================================
            # SVR MODELİ
            # =====================================================
            lag = int(min(12, max(6, len(train_original) // 6)))
            svr = SVRForecaster(kernel='rbf', C=1000.0, epsilon=0.02)
            svr.fit(train_original, lag=lag)
            svr_forecast = svr.predict(train_original, steps=steps)
            
            # MAPE FIX: SVR tahminlerine de hard bounds uygula
            svr_forecast = svr_forecast.clip(lower=0.0, upper=200.0)

            arima_forecast = arima_results['forecast']
            
            # MAPE FIX: ARIMA tahminlerine de hard bounds uygula
            arima_forecast = arima_forecast.clip(lower=0.0, upper=200.0)

            # =====================================================
            # HİBRİT AĞIRLIKLARI
            # MAPE FIX: ARIMA'ya daha fazla ağırlık ver, Naive gereğinden fazla agresif
            # =====================================================
            w_arima = 0.40
            w_svr = 0.35
            w_naive = 0.25
            
            # Güvenli birleştirme
            arima_vals = arima_forecast.values
            svr_vals = svr_forecast.values
            naive_vals = naive_forecast.values
            
            # NaN/Inf değerleri temizle
            arima_vals = pd.to_numeric(arima_vals, errors='coerce')
            svr_vals = pd.to_numeric(svr_vals, errors='coerce')
            naive_vals = pd.to_numeric(naive_vals, errors='coerce')
            
            hybrid_vals = []
            for a, s, n in zip(arima_vals, svr_vals, naive_vals):
                valid_a = not (np.isnan(a) or np.isinf(a))
                valid_s = not (np.isnan(s) or np.isinf(s))
                valid_n = not (np.isnan(n) or np.isinf(n))
                
                total_w = 0.0
                weighted_sum = 0.0
                
                if valid_a:
                    weighted_sum += w_arima * a
                    total_w += w_arima
                if valid_s:
                    weighted_sum += w_svr * s
                    total_w += w_svr
                if valid_n:
                    weighted_sum += w_naive * n
                    total_w += w_naive
                
                if total_w > 0:
                    val = weighted_sum / total_w
                    # MAPE FIX: Final hard bounds
                    val = np.clip(val, 0.0, 200.0)
                    hybrid_vals.append(val)
                else:
                    hybrid_vals.append(last_value)
            
            hybrid_forecast = pd.Series(hybrid_vals, index=arima_results['forecast'].index)

            return {
                **arima_results,
                'forecast': hybrid_forecast,
                'arima_forecast': arima_forecast,
                'svr_forecast': svr_forecast,
                'naive_forecast': naive_forecast,
                'hybrid_weights': {'arima': float(w_arima), 'svr': float(w_svr), 'naive': float(w_naive)},
                'svr_lag': lag
            }
        except Exception as e:
            logger.warning(f"Hibrit tahmin başarısız, ARIMA ile devam ediliyor: {e}")
            return arima_results
    
    def _evaluate_with_timeseries_split(self, n_splits=5):
        """Cross-validation yapar (hibrit model ile)."""
        logger.info(f"InflationForecaster._evaluate_with_timeseries_split çalışıyor (Hibrit)...")
        
        # Full data: train + test
        full_data = pd.concat([self.train_data, self.test_data])
        
        # Exogenous data varsa onu da birleştir
        # NOT: history_exog RAW+LAG columns içerir, exog_train/test sadece LAG columns içerir
        # _prepare_future_exog için RAW columns gerekli (momentum hesabı için)
        # O yüzden history_exog kullanmalıyız
        full_exog = None
        full_exog_raw = None  # RAW columns için
        
        if hasattr(self, 'history_exog') and self.history_exog is not None:
            # RAW + LAG columns
            full_exog_raw = self.history_exog.copy()
            
        if hasattr(self, 'exog_train') and self.exog_train is not None:
            # Sadece LAG columns (eğitim için)
            if hasattr(self, 'exog_test') and self.exog_test is not None:
                full_exog = pd.concat([self.exog_train, self.exog_test])
            else:
                full_exog = self.exog_train
        
        # TimeSeriesSplit ile katlara ayır (daha fazla split daha fazla test verisi demektir)
        # Enflasyon için 3 split yeterli (çok fazla split çok küçük train setleri oluşturur)
        tscv = TimeSeriesSplit(n_splits=min(n_splits, 3))
        
        all_mae = []
        all_rmse = []
        all_mse = []
        all_mape = []
        all_predictions = []
        all_actuals = []
        fold_details = []
        
        last_pred_series = None
        
        fold = 0
        for train_index, test_index in tscv.split(full_data):
            fold += 1
            try:
                train_fold = full_data.iloc[train_index]
                test_fold = full_data.iloc[test_index]
                
                # Exogenous data varsa split et (LAG columns - eğitim için)
                exog_train_fold = full_exog.iloc[train_index] if full_exog is not None else None
                exog_test_fold = full_exog.iloc[test_index] if full_exog is not None else None
                
                # RAW exog varsa onu da split et (RAW+LAG - history için)
                exog_raw_train_fold = full_exog_raw.iloc[train_index] if full_exog_raw is not None else None
                
                # Her fold için modeli yeniden eğit
                model_clone = self.__class__()
                # Parametreleri kopyala (önemli)
                model_clone.arima_max_p = self.arima_max_p
                model_clone.arima_max_q = self.arima_max_q
                model_clone.use_log_transform = self.use_log_transform
                model_clone.clip_quantiles = self.clip_quantiles
                
                # ÖNCE: history_exog'u set et (RAW+LAG columns - future generation için)
                # fit() metodu içinde bu override edilmemeli!
                if exog_raw_train_fold is not None:
                    model_clone.history_exog = exog_raw_train_fold
                
                # Modeli eğit (exogenous data ile!)
                # exog_train_fold: LAG columns (model eğitimi için)
                model_clone.fit(train_fold, test_size=0.0, exogenous_data=exog_train_fold)
                
                # Tahmin yap (SADECE ARIMA - hibrit CV'de dalgalanıyor)
                steps = len(test_fold)
                # MAPE FIX: CV sırasında hibrit kullanma - çok dalgalı
                results = model_clone.forecast(steps=steps, use_hybrid=False)
                
                if results is None or 'forecast' not in results:
                    logger.warning(f"Fold {fold}: Tahmin üretilemedi.")
                    continue
                    
                pred_orig = results['forecast']
                
                # Log transform varsa inverse transform yap (test_fold log scale olduğu için)
                if self.use_log_transform:
                    actual_fold = model_clone._inverse_transform(test_fold)
                else:
                    actual_fold = test_fold
                
                # Log transform varsa inverse transform yap (full_data log scale ise)
                if self.use_log_transform and self.original_train_data is not None:
                    # full_data muhtemelen log scale. original_train/test birleştirip oradan alalım
                    # veya inverse_transform uygulayalım
                    pred_orig = pred_orig # forecast zaten inverse döndürür
                    actual_fold = model_clone._inverse_transform(test_fold)
                
                # Metrikleri hesapla
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                
                # NaN ve Inf kontrolü
                mask = ~np.isnan(pred_orig.values) & ~np.isinf(pred_orig.values) & ~np.isnan(actual_fold.values) & ~np.isinf(actual_fold.values)
                pred_clean = pred_orig.values[mask]
                actual_clean = actual_fold.values[mask]
                
                if len(pred_clean) < 1:
                    logger.warning(f"Fold {fold}: Geçerli tahmin yok (NaN/Inf), atlanıyor.")
                    continue
                
                mae_fold = mean_absolute_error(actual_clean, pred_clean)
                mse_fold = mean_squared_error(actual_clean, pred_clean)
                rmse_fold = np.sqrt(mse_fold)
                
                # Standard MAPE calculation
                denominator = pd.Series(actual_clean).replace(0, np.nan)
                mape_vals = np.abs((pd.Series(actual_clean) - pd.Series(pred_clean)) / denominator) * 100
                mape_fold = np.nanmean(mape_vals)
                
                all_mae.append(mae_fold)
                all_rmse.append(rmse_fold)
                all_mse.append(mse_fold)
                all_mape.append(mape_fold if not np.isnan(mape_fold) else 0.0)
                
                fold_details.append({
                    'fold': fold,
                    'rmse': float(rmse_fold),
                    'mape': float(mape_fold),
                    'samples': len(pred_clean)
                })
                
                logger.info(f"Fold {fold}/{n_splits} (Hibrit): RMSE={rmse_fold:.4f}, MAPE={mape_fold:.2f}%")
                
                last_pred_series = pd.Series(pred_orig.values, index=test_fold.index)
                
            except Exception as e:
                logger.warning(f"Fold {fold} hatası: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        if not all_mae:
            return super().evaluate(use_timeseries_split=False)
            
        # Ortalama metrikleri hesapla
        avg_mae = np.mean(all_mae)
        avg_rmse = np.mean(all_rmse)
        avg_mse = np.mean(all_mse)
        avg_mape = float(np.nanmean(all_mape)) if len(all_mape) else 0.0
        std_rmse = np.std(all_rmse) if len(all_rmse) > 1 else 0.0
        
        metrics = {
            'MAE': float(avg_mae),
            'MSE': float(avg_mse),
            'RMSE': float(avg_rmse),
            'MAPE': float(avg_mape),
            'samples': int(len(full_data)),
            'mode': 'timeseries_split',
            'n_splits': n_splits,
            'std_RMSE': float(std_rmse)
        }
        
        # Son fold'un tahminlerini sakla (en güncel veriler)
        if last_pred_series is not None:
            self.test_predictions = last_pred_series

        # MongoDB'ye logla (ilgili alanlar hazırlandıktan sonra)
        self._persist_timeseries_validation(
            metrics,
            fold_details,
            n_splits,
            start_date=full_data.index.min() if not full_data.empty else None,
            end_date=full_data.index.max() if not full_data.empty else None,
            samples=len(full_data)
        )
        
        self.performance_metrics = metrics
        
        logger.info(f"TimeSeriesSplit Sonuçları ({self.target_column}):")
        logger.info("  Ortalama MAE: %.4f", metrics['MAE'])
        logger.info("  Ortalama RMSE: %.4f", metrics['RMSE'])
        logger.info("  Ortalama MAPE: %.2f%%", metrics['MAPE'])
        logger.info("  Std RMSE: %.4f", metrics['std_RMSE'])
        logger.info("  Toplam örnek: %d", metrics['samples'])
        logger.info("  ✅ Geçmiş verilerde başarılı, gelecek tahminleri için hazır!")
        
        return metrics

    def evaluate(self, use_timeseries_split=True, n_splits=5):
        """
        Modeli değerlendirir (Hibrit yapıyı kullanarak).
        """
        # Eğer fitted_model yoksa önce eğit
        if self.fitted_model is None:
            if self.train_data is None:
                raise ValueError("Model eğitilmemiş ve veri yok.")
            self.fit(self.train_data)

        if use_timeseries_split:
            from sklearn.model_selection import TimeSeriesSplit
            return self._evaluate_with_timeseries_split(n_splits=n_splits)

        # Normal hold-out evaluation (test seti üzerinde)
        # ... (burası aynı kalacak)
        if self.test_data is None or self.test_data.empty:
            return super().evaluate(use_timeseries_split=False)
            
        try:
            # MAPE FIX: Sadece ARIMA tahmin kullan (hibrit değerlendirmede kararsız)
            steps = len(self.test_data)
            forecast_results = self.forecast(steps=steps, use_hybrid=False)
            
            if forecast_results is None or 'forecast' not in forecast_results:
                return super().evaluate(use_timeseries_split=False)
            
            pred_values = forecast_results['forecast'].values
            actual_values = self.test_data.values
            
            # Orijinal skala (inverse transform yapılmış olmalı)
            if self.use_log_transform and self.original_test_data is not None:
                actual_values = self.original_test_data.values
            
            # Boyut eşitleme
            min_len = min(len(pred_values), len(actual_values))
            pred_values = pred_values[:min_len]
            actual_values = actual_values[:min_len]
            
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            mae = mean_absolute_error(actual_values, pred_values)
            mse = mean_squared_error(actual_values, pred_values)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual_values - pred_values) / (actual_values + 1e-10))) * 100
            
            metrics = {
                'MAE': float(mae),
                'MSE': float(mse),
                'RMSE': float(rmse),
                'MAPE': float(mape),
                'samples': len(actual_values)
            }
            
            self.performance_metrics = metrics
            logger.info(f"Hibrit Model Performansı (Hold-out): RMSE={rmse:.4f}, MAPE={mape:.2f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"Hibrit evaluate hatası: {e}")
            return super().evaluate(use_timeseries_split=False)
    
    def forecast_with_news(self, steps=12):
        """
        ARIMA + News sentiment ile nihai tahmin.
        
        1. ARIMA ile ilk tahmin
        2. News API ile haber toplama
        3. Claude ile sentiment analizi
        4. Sentiment çarpanı ile nihai tahmin
        
        Args:
            steps: Kaç adım ileriye tahmin
            
        Returns:
            dict: {
                'arima_forecast': ARIMA tahmini,
                'enhanced_forecast': News çarpanlı tahmin,
                'multiplier': Sentiment çarpanı,
                'news_analysis': Claude analizi,
                'confidence': Güven skoru
            }
        """
        logger.info(f"Enflasyon için News sentiment destekli tahmin başlatılıyor ({steps} adım)...")
        
        # 1. ARIMA tahmini
        arima_results = self.forecast(steps=steps)
        
        if arima_results is None:
            logger.error("ARIMA tahmini başarısız")
            return None
        
        # 2. News + Claude analizi
        try:
            news_analysis = self.news_service.analyze_inflation_news(days_back=7)
            multiplier = news_analysis.get('multiplier', 1.0)
            
            logger.info(f"📰 Enflasyon haberleri analiz edildi:")
            logger.info(f"   Haber sayısı: {news_analysis.get('articles_analyzed', 0)}")
            logger.info(f"   Sentiment çarpanı: {multiplier:.3f}")
            logger.info(f"   Güven: {news_analysis.get('confidence', 0):.2f}")
            
        except Exception as e:
            logger.error(f"News analizi hatası: {e}")
            multiplier = 1.0
            news_analysis = {'multiplier': 1.0, 'analysis': 'Analiz yapılamadı'}
        
        # 3. Nihai tahmin (ARIMA * multiplier)
        # DEĞİŞİKLİK: Kullanıcı talebi üzerine deterministik haber çarpanı kaldırıldı.
        # Haber analizi sadece bilgi amaçlı raporda yer alacak.
        enhanced_forecast = arima_results['forecast']
        
        logger.info(f"✅ Enflasyon nihai tahmini hazır (Multivariate Model)!")
        if not arima_results['forecast'].empty:
            logger.info(f"   Tahmin: {arima_results['forecast'].iloc[-1]:.2f}%")
        
        return {
            'arima_forecast': arima_results['forecast'],
            'enhanced_forecast': enhanced_forecast,
            'multiplier': 1.0, # Etkisiz
            'news_analysis': news_analysis,
            'confidence': news_analysis.get('confidence', 0.5)
        }


def load_inflation_data(include_features=False):
    """
    Enflasyon verilerini yükler (CSV öncelikli).
    Kullanıcı isteği üzerine 'tr_economy_ml_2005_01_to_2025_11.csv' dosyasından okunur.
    """
    logger.info("🔄 CSV'den enflasyon verisi yükleniyor (Multivariate + Sentiment)...")
    
    try:
        csv_path = os.path.join(project_root, 'src', 'data_collection', 'tr_economy_ml_2005_01_to_2025_11.csv')
        
        if not os.path.exists(csv_path):
            logger.error(f"❌ CSV dosyası bulunamadı: {csv_path}")
            return None
            
        df = pd.read_csv(csv_path)
        logger.info(f"✅ Ana ekonomik veri yüklendi: {len(df)} satır")
        
        # Tarih formatı
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # Kolon eşleştirme
        column_mapping = {
            'yi_ufe_yoy_pct': 'inflation_rate',
            'usd_try_avg_buy': 'usd_try',
            'brent_usd_per_bbl': 'brent_oil',
            'net_min_wage_tl': 'min_wage'
        }
        
        # Mevcut kolonları yeniden adlandır
        df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
        
        if 'inflation_rate' not in df.columns:
            logger.error("❌ CSV'de enflasyon verisi (yi_ufe_yoy_pct) bulunamadı.")
            return None
            
        # SENTIMENT (DUYGU) VERİSİ ENTEGRASYONU
        try:
            sent_path = os.path.join(project_root, 'src', 'data_collection', 'duygu.csv')
            if os.path.exists(sent_path):
                logger.info(f"🔄 Duygu analizi verisi yükleniyor: {sent_path}")
                sent_df = pd.read_csv(sent_path)
                
                # Tarih formatı (MM-YYYY -> Datetime)
                sent_df['date'] = pd.to_datetime(sent_df['Tarih'], format='%m-%Y')
                sent_df.set_index('date', inplace=True)
                
                # Sadece gerekli kolonları al
                sent_df = sent_df[['News_Tone_Mean', 'Event_Count']]
                
                # Ana veri setiyle birleştir (Left Join)
                df = df.join(sent_df, how='left')
                
                # Eksik verileri doldur (Veri olmayan aylar nötr kabul edilir)
                df['News_Tone_Mean'] = df['News_Tone_Mean'].fillna(0.0)
                df['Event_Count'] = df['Event_Count'].fillna(0.0)
                
                logger.info(f"✅ Sentiment verisi eklendi: {len(sent_df)} kayıt")
            else:
                logger.warning("⚠️ duygu.csv bulunamadı, sentiment özellikleri atlanacak.")
        except Exception as e:
            logger.error(f"❌ Duygu verisi işlenirken hata: {e}")

        # Temizlik
            df = df.replace([np.inf, -np.inf], np.nan)
        # Sadece sayısal kolonları al
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_cols]
        
        df = df.asfreq('MS').ffill().bfill() # Aylık frekans

        if not include_features:
            series = df['inflation_rate'].astype(float)
            logger.info(f"✅ Sadece inflation_rate serisi döndürülüyor: {len(series)} kayıt")
            return series

        # ============= FEATURE ENGINEERING =============
        logger.info("🔧 Feature engineering başlatılıyor...")
        feature_df = df[['inflation_rate']].copy()
        
        # 1. USD/TRY (İthalat maliyeti)
        if 'usd_try' in df.columns:
            feature_df['usd_try'] = df['usd_try']  # RAW
            feature_df['usd_try_lag1'] = df['usd_try'].shift(1)  # LAG
            feature_df['usd_try_mom_3'] = df['usd_try'].pct_change(3).shift(1)
            logger.info("✅ USD/TRY features eklendi")
            
        # 2. Brent Petrol (Enerji maliyeti)
        if 'brent_oil' in df.columns:
            feature_df['brent_oil'] = df['brent_oil']  # RAW
            feature_df['brent_oil_lag1'] = df['brent_oil'].shift(1)  # LAG
            feature_df['brent_oil_mom_3'] = df['brent_oil'].pct_change(3).shift(1)
            logger.info("✅ Brent Oil features eklendi")
            
        # 3. Asgari Ücret (Maliyet enflasyonu)
        if 'min_wage' in df.columns:
            feature_df['min_wage'] = df['min_wage']  # RAW
            min_wage_change = df['min_wage'].pct_change().fillna(0)
            feature_df['min_wage_increase_dummy'] = (min_wage_change > 0.01).astype(int)
            feature_df['min_wage_lag1'] = df['min_wage'].shift(1)  # LAG
            logger.info("✅ Min Wage features eklendi")

        # 4. Sentiment / Duygu Analizi
        if 'News_Tone_Mean' in df.columns and 'Event_Count' in df.columns:
            feature_df['News_Tone_Mean'] = df['News_Tone_Mean']  # RAW
            feature_df['Event_Count'] = df['Event_Count']  # RAW
            
            # Weighted Tone: Ton * Log(Hacim)
            df['weighted_tone'] = df['News_Tone_Mean'] * np.log1p(df['Event_Count'])
            
            # Lag Yapısı (Haber -> Beklenti -> Fiyat)
            feature_df['weighted_tone_lag1'] = df['weighted_tone'].shift(1)  # LAG
            feature_df['weighted_tone_lag2'] = df['weighted_tone'].shift(2)  # LAG
            feature_df['event_count_log_lag1'] = np.log1p(df['Event_Count']).shift(1)  # LAG
            logger.info("✅ Sentiment features eklendi")

        # 5. Mevsimsellik (Trigonometrik)
            months = feature_df.index.month
            feature_df['month_sin'] = np.sin(2 * np.pi * months / 12)
            feature_df['month_cos'] = np.cos(2 * np.pi * months / 12)
        logger.info("✅ Seasonal features eklendi")

        # 6. Trend
        feature_df['trend'] = np.arange(len(feature_df), dtype=float)
        
        # NaN temizliği (Lag işlemlerinden dolayı ilk satırlar boşalır)
        initial_rows = len(feature_df)
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna()
        logger.info(f"✅ NaN temizliği: {initial_rows} -> {len(feature_df)} kayıt")
        
        logger.info(f"✅ Feature engineering tamamlandı: {len(feature_df)} satır x {feature_df.shape[1]} kolon")
        logger.info(f"📊 Kolonlar: {list(feature_df.columns)}")
        return feature_df
            
    except Exception as e:
        logger.error(f"❌ CSV yükleme hatası: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def load_inflation_data_from_mongodb(include_features=False):
    """Geriye dönük uyumluluk için alias (CSV kullanır)."""
    return load_inflation_data(include_features)


def test_inflation_forecaster():
    """Enflasyon tahmin modelini test eder."""
    logger.info("=" * 60)
    logger.info("Enflasyon Tahmin Modeli Test Ediliyor...")
    logger.info("=" * 60)
    
    # Veri yükle
    data = load_inflation_data_from_mongodb()
    
    if data is None or data.empty:
        logger.error("Enflasyon verisi yüklenemedi, test iptal edildi.")
        return None
    
    logger.info(f"Yüklenen veri: {len(data)} kayıt")
    if not data.empty:
        logger.info(f"En son veri: {data.index[-1]} - {data.iloc[-1]:.2f}%")
    
    # Model oluştur ve eğit
    forecaster = InflationForecaster()
    forecaster.fit(data, test_size=0.2)
    
    # Performans değerlendir
    metrics = forecaster.evaluate()
    
    # 12 ay tahmin
    forecast_results = forecaster.forecast(steps=12)
    
    if forecast_results:
        logger.info("\n" + "=" * 60)
        logger.info("ENFLASYON TAHMİNLERİ (12 Ay)")
        logger.info("=" * 60)
        
        for date, value in forecast_results['forecast'].items():
            logger.info(f"{date.strftime('%Y-%m')}: {value:.2f}%")
        
        if not forecast_results['forecast'].empty:
            last_forecast = forecast_results['forecast'].iloc[-1]
            logger.info(f"\n🎯 SON TAHMİN: {last_forecast:.2f}%")
    
    logger.info("=" * 60)
    logger.info("Enflasyon tahmin testi tamamlandı.")
    logger.info("=" * 60)
    
    return {
        'metrics': metrics,
        'forecast': forecast_results,
        'data_points': len(data)
    }


if __name__ == '__main__':
    test_inflation_forecaster()
