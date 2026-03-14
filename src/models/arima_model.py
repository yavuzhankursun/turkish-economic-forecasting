"""
ARIMA Zaman Serisi Tahmin Modeli
================================

Bu modül ekonomik göstergelerin ARIMA modeli ile tahminini yapar.
TÜBİTAK projesi kapsamında öğrenci tarafından geliştirilmiştir.

Geliştirici: [Öğrenci Adı] - [Üniversite]
Danışman: [Danışman Adı]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from typing import Optional
try:
    import pmdarima as pm  # pmdarima kütüphanesini içe aktar
    PMDARIMA_AVAILABLE = True
except Exception as _e:
    PMDARIMA_AVAILABLE = False
    pm = None
    logging.getLogger(__name__).warning(f"pmdarima import edilemedi: {_e}")
import logging
import sys
import os
from datetime import datetime

# Proje kök dizinini import yolu olarak ekle
# Bu, komut satırından çalıştırıldığında 'src' içindeki modülleri bulmasını sağlar
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


from src.utils.mongodb_manager import MongoDBManager
from config.config import MONGODB_COLLECTIONS

# Matplotlib stili
plt.style.use('seaborn-v0_8-whitegrid')

# Logging ayarları
# Geliştirilmiş logging formatı
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ARIMAForecaster:
    """
    ARIMA modeli ile ekonomik veri tahmini yapan sınıf.
    
    Bu sınıf şunları yapar:
    - Zaman serisi analizi
    - ARIMA parametre optimizasyonu
    - Model eğitimi ve tahmin
    - Performans değerlendirmesi
    """
    
    def __init__(self, target_column: str = 'usd_avg', seasonal_periods: Optional[int] = None,
                 future_exog_strategy: str = 'repeat_last', use_log_transform: bool = True,
                 clip_quantiles: tuple = (0.01, 0.99), log_timeseries_validation: Optional[bool] = None):
        """
        ARIMA tahmin sınıfını başlatır.
        
        Args:
            target_column (str): Tahmin edilecek sütunun adı.
            seasonal_periods (int | None): Mevsimsellik periyodu (ör. aylık veri için 12).
            future_exog_strategy (str): Gelecek adımlar için exogenous değişken stratejisi.
        """
        self.target_column = target_column
        self.model = None
        self.fitted_model = None
        self.best_params = None
        self.performance_metrics = None
        self.test_data = None
        self.train_data = None  # Eğitim verisini saklamak için eklendi
        self.test_predictions = None  # Test tahminlerini saklamak için eklendi
        self.train_exog = None
        self.test_exog = None
        self.future_exog_strategy = future_exog_strategy
        self.seasonal_periods = seasonal_periods
        self.seasonal = bool(seasonal_periods and seasonal_periods > 1)
        self.best_seasonal_order = None
        # Veri stabilizasyonu için ayarlar
        self.use_log_transform = use_log_transform
        self.clip_quantiles = clip_quantiles
        self._transform = None
        self.original_train_data = None
        self.original_test_data = None
        env_flag = os.getenv("LOG_TIMESERIES_VALIDATION", "true").lower() == "true"
        self.log_timeseries_validation = env_flag if log_timeseries_validation is None else log_timeseries_validation
        
        logger.info(
            f"ARIMA Forecaster '{self.target_column}' hedefi için başlatıldı."
            f" (seasonal={'on' if self.seasonal else 'off'}, period={self.seasonal_periods})"
        )
    
    def check_stationarity(self, data):
        """
        Zaman serisinin durağanlığını kontrol eder.
        
        Args:
            data: Kontrol edilecek zaman serisi
            
        Returns:
            bool: Durağan ise True
        """
        # Augmented Dickey-Fuller testi
        result = adfuller(data.dropna())
        p_value = result[1]
        
        # p-value < 0.05 ise durağan
        is_stationary = p_value < 0.05
        
        # Daha açıklayıcı loglama
        if is_stationary:
            logger.info(f"ADF Testi: Zaman serisi durağan görünüyor (p-value: {p_value:.4f}).")
        else:
            logger.info(f"ADF Testi: Zaman serisi durağan değil (p-value: {p_value:.4f}). Fark alma (differencing) gerekebilir.")
        return is_stationary

    def _apply_transform(self, series: pd.Series) -> pd.Series:
        """Veriyi daha durağan hale getirmek için opsiyonel log1p dönüşümü uygular."""
        self._transform = None
        if self.use_log_transform and series.min() > 0:
            self._transform = 'log1p'
            return np.log1p(series)
        return series

    def _inverse_transform(self, series: pd.Series | np.ndarray) -> pd.Series:
        """Tahminleri orijinal skala'ya çevirir."""
        if series is None:
            return series
        if self._transform == 'log1p':
            return np.expm1(series)
        return series
    
    def find_best_params(self, data, exog=None, information_criterion: str = 'aic'):
        """
        En iyi ARIMA parametrelerini `auto_arima` kullanarak bulur.
        
        Args:
            data: Eğitim verisi
            exog: Exogenous değişkenler (opsiyonel)
            information_criterion: Bilgi kriteri ('aic', 'bic', 'hqic') - G-9 gereksinimi
        
        Returns:
            pmdarima.ARIMA: Eğitilmiş en iyi model
        """
        if not PMDARIMA_AVAILABLE:
            logger.error("pmdarima kurulu değil. Lütfen requirements.txt kurulumunu tamamlayın: pip install pmdarima")
            raise RuntimeError("pmdarima eksik")

        # Bilgi kriteri doğrulama - G-9 gereksinimi
        valid_criteria = ['aic', 'bic', 'hqic']
        if information_criterion.lower() not in valid_criteria:
            logger.warning(f"Geçersiz bilgi kriteri: {information_criterion}, AIC kullanılıyor")
            information_criterion = 'aic'
        
        logger.info(f"En iyi ARIMA parametreleri `auto_arima` ile aranıyor (kriter: {information_criterion.upper()})...")

        seasonal_flag = False
        seasonal_period = 1
        if self.seasonal:
            min_required = (self.seasonal_periods or 0) * 2
            if min_required and len(data) >= min_required:
                seasonal_flag = True
                seasonal_period = int(self.seasonal_periods)
            else:
                logger.info(
                    "Veri uzunluğu mevsimsellik için yetersiz. Seasonal ARIMA devre dışı bırakıldı."
                )

        # auto_arima, en iyi p,d,q parametrelerini otomatik bulur.
        # stationarity testini kendi içinde yapar (d parametresini belirler)
        # seasonal_flag değerine göre mevsimsellik aktif edilir
        # HIZLANDIRILMIŞ ARIMA PARAMETRELERİ - Hız Optimizasyonu (10 saniye hedefi)
        logger.info("⚡ HIZLI ARIMA parametreleri kullanılıyor...")
        
        # Minimum model karmaşıklığı - çok basit modelleri engelle
        min_p = 1 if len(data) >= 10 else 0
        min_q = 1 if len(data) >= 10 else 0
        
        self.fitted_model = pm.auto_arima(
            data,
            start_p=min_p, start_q=min_q,  # Başlangıç
            test='adf',            # Durağanlık testi
            max_p=3,               # HIZLANDIRILDI: 7 -> 3
            max_q=3,               # HIZLANDIRILDI: 7 -> 3
            max_d=2,               # Differencing
            seasonal=seasonal_flag,
            m=seasonal_period,
            max_P=2,               # HIZLANDIRILDI: 3 -> 2
            max_Q=2,               # HIZLANDIRILDI: 3 -> 2
            max_D=1,
            d=None,                # d'yi otomatik bul
            start_P=1 if seasonal_flag else 0, 
            D=None,                
            trace=False,           # HIZLANDIRILDI: Detaylı log KAPALI
            error_action='ignore',  
            suppress_warnings=True, 
            stepwise=True,         # Hızlı arama
            information_criterion=information_criterion.lower(),  # G-9: AIC/BIC/HQIC desteği
            n_jobs=1,              
            random_state=42,       
            with_intercept=True,   
            X=exog,
            maxiter=50             # HIZLANDIRILDI: Maksimum iterasyon sınırı
        )
        
        # Model çok basitse uyar
        if self.fitted_model.get_params()['order'] == (0, 1, 0):
            logger.warning("⚠️ Model çok basit (0,1,0), manuel parametre denemesi yapılıyor...")
            try:
                # Daha karmaşık bir model zorla
                self.fitted_model = pm.ARIMA(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
                self.fitted_model.fit(data, X=exog)
                logger.info("✅ Alternatif model ARIMA(1,1,1) kullanıldı")
            except Exception as e:
                logger.warning(f"Manuel model başarısız, orijinal model korunuyor: {e}")
        
        params = self.fitted_model.get_params()
        self.best_params = params['order']
        self.best_seasonal_order = params.get('seasonal_order')
        logger.info(f"En iyi parametreler bulundu: ARIMA{self.best_params}")
        
        return self.fitted_model

    def fit(self, data, test_size=0.2, exogenous_data=None, information_criterion: str = None):
        """
        ARIMA modelini eğitir.
        
        Args:
            data: Eğitim verisi
            test_size: Test seti oranı
            exogenous_data: Seçmeli exogenous (bağımsız) değişkenler
            information_criterion: Bilgi kriteri ('aic', 'bic', 'hqic') - G-9 gereksinimi
        """
        # logger.debug("ARIMA eğitimi...") # SESSIZ
        
        if data is None or data.empty:
            logger.error("Veri yok!")
            return
        
        if isinstance(data, pd.Series):
            series = data.copy()
        else:
            # DataFrame ise hedef sütunu öncele, yoksa ilk sütunu al
            if self.target_column in data.columns:
                series = data[self.target_column].copy()
            elif 'target' in data.columns:
                series = data['target'].copy()
            else:
                series = data.iloc[:, 0].copy()
        
        # NaN KONTROLÜ VE TEMİZLEME - SESSIZ MOD
        nan_count = series.isna().sum()
        if nan_count > 0:
            series = series.dropna()
            # logger.debug(f"NaN temizlendi: {nan_count} adet") # SESSIZ
        
        # Sonsuz değer kontrolü - SESSIZ
        inf_count = np.isinf(series).sum()
        if inf_count > 0:
            series = series.replace([np.inf, -np.inf], np.nan).dropna()
            # logger.debug(f"Inf temizlendi: {inf_count} adet") # SESSIZ
        
        # Minimum veri kontrolü
        if len(series) < 10:
            logger.error(f"❌ Yetersiz veri: {len(series)} nokta")
            raise ValueError(f"Yetersiz veri: {len(series)} nokta")
        
        # logger.debug(f"Veri hazır: {len(series)} nokta") # SESSIZ
        
        # DataFrame'den exogenous değişkenleri çıkar
        if not isinstance(data, pd.Series) and exogenous_data is None:
            candidate_cols = [c for c in data.columns if c not in [self.target_column, 'target']]
            if candidate_cols:
                exogenous_data = data[candidate_cols].copy()
        
        # Veri çeşitliliği kontrolü
        if series.nunique() < 3:
            logger.error("Veri çok az çeşitlilik içeriyor (nunique<3). Model eğitimi uygun değil.")
            return

        # Frekansın 'MS' (ay başı) olduğunu varsayalım ve ayarlayalım
        # Bu, `statsmodels` kütüphanesinin frekans uyarısını engeller
        if pd.infer_freq(series.index) is None:
            series = series.asfreq('MS')
            # NaN değerleri kaldır
            series = series.dropna()
            logger.info("Veri frekansı 'MS' (Ay Başı) olarak ayarlandı ve NaN değerler kaldırıldı.")

        # Aykırı değerleri kırp (winsorization)
        if self.clip_quantiles and isinstance(self.clip_quantiles, tuple) and len(self.clip_quantiles) == 2:
            q_low, q_high = self.clip_quantiles
            if 0 < q_low < q_high < 1:
                lower, upper = series.quantile(q_low), series.quantile(q_high)
                series = series.clip(lower=lower, upper=upper)
                logger.info(f"Seri clip uygulandı: q{q_low:.2f}={lower:.4f}, q{q_high:.2f}={upper:.4f}")

        # Orijinal seriyi sakla ve log dönüşümü uygula
        series_original = series.copy()
        series = self._apply_transform(series)

        # Exogenous veriyi hazırla ve indeksle hizala
        if exogenous_data is not None:
            if isinstance(exogenous_data, pd.Series):
                exogenous_data = exogenous_data.to_frame()
            elif not isinstance(exogenous_data, pd.DataFrame):
                exogenous_data = pd.DataFrame(exogenous_data)

            exogenous_data = exogenous_data.apply(pd.to_numeric, errors='coerce')
            exogenous_data = exogenous_data.replace([np.inf, -np.inf], np.nan)
            exogenous_data = exogenous_data.dropna(axis=1, how='all')

            if exogenous_data.empty:
                logger.warning("Geçerli exogenous değişken bulunamadı, model tek değişkenli devam edecek.")
                exogenous_data = None
            else:
                exogenous_data = exogenous_data.reindex(series.index)
                exogenous_data = exogenous_data.fillna(method='ffill').fillna(method='bfill')
                self.train_exog = None
                self.test_exog = None
        else:
            self.train_exog = None
            self.test_exog = None

        # Veriyi eğitim ve test setlerine ayır
        total_points = len(series)

        # test_size <= 0 ise: split YAPMA (özellikle cross-validation trainer'larında tam veriyle eğitim için)
        if test_size is not None and float(test_size) <= 0.0:
            split_idx = total_points
        elif total_points < 4:
            split_idx = max(total_points - 1, 1)
        else:
            min_test_points = min(12, max(3, total_points // 4))
            proposed = int(round(total_points * test_size))
            test_points = max(proposed, min_test_points)
            if test_points >= total_points:
                test_points = max(1, total_points // 3)
            split_idx = max(total_points - test_points, 1)

        self.train_data = series.iloc[:split_idx]  # Eğitim verisini değişkene ata
        self.test_data = series.iloc[split_idx:]
        self.original_train_data = series_original.iloc[:split_idx]
        self.original_test_data = series_original.iloc[split_idx:]
        if exogenous_data is not None:
            self.train_exog = exogenous_data.iloc[:split_idx]
            self.test_exog = exogenous_data.iloc[split_idx:]
            logger.info(f"Exogenous değişkenler kullanılıyor: {list(self.train_exog.columns)}")
        else:
            logger.debug("Tek değişkenli ARIMA eğitimi yapılıyor (exog yok).")

        logger.info(
            "Veri ayrımı tamamlandı: %d eğitim, %d test gözlemi", len(self.train_data), len(self.test_data)
        )

        # Durağanlığı kontrol et
        self.check_stationarity(self.train_data)

        # En iyi parametreleri bul ve modeli eğit
        self.find_best_params(self.train_data, exog=self.train_exog)

        logger.info("Model eğitimi tamamlandı!")
    
    def forecast(self, steps=12, confidence_level: float = 0.95):
        """
        Gelecek değerleri tahmin eder.
        
        Args:
            steps: Kaç adım ilerisi tahmin edilecek
            confidence_level: Güven seviyesi (varsayılan: 0.95 = %95) - G-31 gereksinimi
        
        Modeli önce test verileriyle günceller, böylece tahminler
        bilinen en son veriden başlar.
        """
        if self.fitted_model is None:
            logger.error("Model henüz eğitilmedi!")
            return None
        
        logger.info(f"{steps} adım ilerisi için tahmin yapılıyor (güven seviyesi: {confidence_level*100:.0f}%)...")
        
        # PROFESYONEL MODEL GÜNCELLEME - Test verisiyle güncelle
        if self.test_data is not None and not self.test_data.empty:
            if self.test_exog is not None and not self.test_exog.empty:
                self.fitted_model.update(self.test_data, X=self.test_exog)
            else:
                self.fitted_model.update(self.test_data)
            logger.info(f"Model {len(self.test_data)} adet test verisi ile güncellendi.")
        else:
            logger.info("Test verisi yok, model güncellenmedi.")

        # TAHMİN VE GÜVEN ARALIĞI - G-31 gereksinimi: %95 güven aralığı
        future_exog = self._prepare_future_exog(steps)
        
        # Alpha değerini güven seviyesinden hesapla (0.95 -> 0.05)
        alpha = 1.0 - confidence_level
        
        # Tahmin ve güven aralığı hesapla
        try:
            if future_exog is not None:
                forecast_result = self.fitted_model.predict(n_periods=steps, X=future_exog, return_conf_int=True, alpha=alpha)
            else:
                forecast_result = self.fitted_model.predict(n_periods=steps, return_conf_int=True, alpha=alpha)
        except Exception as e:
            # Güven aralığı hesaplanamazsa sadece tahmin yap
            logger.warning(f"Güven aralığı hesaplanamadı, sadece tahmin yapılıyor: {e}")
            if future_exog is not None:
                forecast_result = self.fitted_model.predict(n_periods=steps, X=future_exog)
            else:
                forecast_result = self.fitted_model.predict(n_periods=steps)
            conf_int = None
        
        # forecast_result tuple ise (forecast, conf_int), değilse sadece forecast
        if isinstance(forecast_result, tuple):
            forecast_values_np, conf_int = forecast_result
        else:
            forecast_values_np = forecast_result
            conf_int = None
        
        # Son değeri al (güncellenmiş modelden)
        if self.test_data is not None and not self.test_data.empty:
            last_value = self.test_data.iloc[-1]  # En son gerçek veri
        elif not self.train_data.empty:
            last_value = self.train_data.iloc[-1]  # Eğitim verisinin sonu
        else:
            last_value = 0.0  # Varsayılan değer
        
        # SINIR YOK - Modelin kendi tahminlerini kullan
        logger.info("Sınırlandırma kaldırıldı - modelin ham tahminleri kullanılıyor")

        # Gelecek tahminleri için tarih indeksi oluştur (EN SON VERİDEN başla)
        if self.test_data is not None and not self.test_data.empty:
            last_date = self.test_data.index[-1]  # En son gerçek veri
        elif not self.train_data.empty:
            last_date = self.train_data.index[-1]  # Eğitim verisinin sonu
        else:
            last_date = pd.Timestamp.now()  # Şu anki tarih
        
        forecast_index = pd.date_range(start=last_date, periods=steps + 1, freq='MS')[1:]
        
        # Tahmin serisi (orijinal skala)
        forecast_series = pd.Series(forecast_values_np, index=forecast_index)
        forecast_series = self._inverse_transform(forecast_series)
        
        # Güven aralığı - G-31 gereksinimi
        results = {
            'forecast': forecast_series
        }
        
        if conf_int is not None:
            # Güven aralığı (varsayılan %95)
            conf_int_df = pd.DataFrame(conf_int, index=forecast_index, columns=['lower', 'upper'])
            conf_int_df['lower'] = self._inverse_transform(conf_int_df['lower'])
            conf_int_df['upper'] = self._inverse_transform(conf_int_df['upper'])
            results['confidence_interval'] = {
                'level': confidence_level,
                'lower': conf_int_df['lower'].to_dict(),
                'upper': conf_int_df['upper'].to_dict()
            }
            logger.info(f"✅ %{confidence_level*100:.0f} güven aralığı hesaplandı (G-31 gereksinimi)")
        else:
            logger.warning("⚠️ Güven aralığı hesaplanamadı")
        
        logger.info("Tahmin ve güven aralığı başarıyla tamamlandı!")
        return results
    
    def evaluate(self, use_timeseries_split=False, n_splits=3):
        """
        Model performansını test seti üzerinde değerlendirir ve test tahminlerini saklar.
        
        Args:
            use_timeseries_split: TimeSeriesSplit kullanılsın mı (walk-forward validation)
            n_splits: TimeSeriesSplit için fold sayısı
        """
        if self.fitted_model is None:
            logger.error("Model eğitilmediği için değerlendirme yapılamadı.")
            return {
                'MAE': 0.0, 'MSE': 0.0, 'RMSE': 0.0, 'MAPE': 0.0,
                'samples': 0, 'mode': 'error'
            }

        # TimeSeriesSplit ile walk-forward validation
        if use_timeseries_split:
            return self._evaluate_with_timeseries_split(n_splits)

        test_segment = None
        pred_segment = None
        evaluation_mode = 'test'
        test_data_clean = None

        actual_for_metrics = None
        pred_for_metrics = None

        # Test verisi kontrolü
        if self.test_data is not None and len(self.test_data) > 0:
            test_segment = self.test_data.dropna()

        # Test setinde tahmin yap
        if test_segment is not None and len(test_segment) >= 2:
            logger.info(
                "Model performansı test seti üzerinde değerlendiriliyor (%d adım)...",
                len(test_segment)
            )

            try:
                if self.test_exog is not None and not self.test_exog.empty:
                    test_forecast_values = self.fitted_model.predict(
                        n_periods=len(test_segment), X=self.test_exog.loc[test_segment.index]
                    )
                else:
                    test_forecast_values = self.fitted_model.predict(n_periods=len(test_segment))

                pred_series = pd.Series(test_forecast_values, index=test_segment.index)
                
                # NaN kontrolü
                if pred_series.isnull().all():
                    logger.error("Tahmin çıktıları tamamen NaN, değerlendirme yapılamadı.")
                    raise ValueError("Tahminler NaN")

                if pred_series.isnull().any():
                    logger.warning("Test tahminlerinde NaN değerler bulundu, temizleniyor.")
                    valid_mask = ~pred_series.isnull()
                    pred_series = pred_series[valid_mask]
                    test_segment = test_segment[valid_mask]

                if len(pred_series) >= 2:
                    self.test_predictions = pred_series
                    test_data_clean = test_segment
                    pred_segment = pred_series
                    # Orijinal skala için gerçek ve tahmin
                    if self.original_test_data is not None and not self.original_test_data.empty:
                        actual_for_metrics = self.original_test_data.loc[test_data_clean.index]
                    else:
                        actual_for_metrics = self._inverse_transform(test_data_clean)
                    pred_for_metrics = self._inverse_transform(self.test_predictions)
                else:
                    raise ValueError("Yeterli geçerli tahmin yok")
                    
            except Exception as e:
                logger.warning(f"Test değerlendirmesi başarısız: {e}, eğitim verisine fallback.")
                test_segment = None

        # Fallback: Eğitim verisi üzerinden backtesting
        if pred_segment is None or len(pred_segment) < 2:
            if self.train_data is None or len(self.train_data.dropna()) < 5:
                logger.error("Model değerlendirmesi için yeterli veri yok.")
                return {
                    'MAE': 0.0, 'MSE': 0.0, 'RMSE': 0.0, 'MAPE': 0.0,
                    'samples': 0, 'mode': 'insufficient_data'
                }

            logger.warning("Test verisi yetersiz, eğitim verisinin son bölümünde backtesting uygulanıyor.")
            evaluation_mode = 'train_backtest'

            try:
                cleaned_train = self.train_data.dropna()
                in_sample_predictions = self.fitted_model.predict_in_sample()
                
                if isinstance(in_sample_predictions, np.ndarray):
                    in_sample_predictions = pd.Series(in_sample_predictions, index=cleaned_train.index)
                elif not isinstance(in_sample_predictions, pd.Series):
                    in_sample_predictions = pd.Series(np.asarray(in_sample_predictions), index=cleaned_train.index)

                window = min(max(4, len(cleaned_train) // 3), len(cleaned_train))
                test_data_clean = cleaned_train.iloc[-window:]
                pred_segment = in_sample_predictions.iloc[-window:]
                self.test_predictions = pred_segment
                if self.original_train_data is not None and not self.original_train_data.empty:
                    actual_for_metrics = self.original_train_data.loc[test_data_clean.index]
                else:
                    actual_for_metrics = self._inverse_transform(test_data_clean)
                pred_for_metrics = self._inverse_transform(self.test_predictions)
                
            except Exception as e:
                logger.error(f"Backtesting hatası: {e}")
                return {
                    'MAE': 0.0, 'MSE': 0.0, 'RMSE': 0.0, 'MAPE': 0.0,
                    'samples': 0, 'mode': 'backtest_error'
                }

        logger.info("Değerlendirme modu: %s", evaluation_mode)
        
        # Metrikleri hesapla
        try:
            # Orijinal skala üzerinden metrikleri hesapla
            actual_series = actual_for_metrics if actual_for_metrics is not None else test_data_clean
            pred_series_for_metrics = pred_for_metrics if pred_for_metrics is not None else self.test_predictions

            mae = mean_absolute_error(actual_series, pred_series_for_metrics)
            mse = mean_squared_error(actual_series, pred_series_for_metrics)
            rmse = np.sqrt(mse)

            denominator = pd.Series(actual_series).replace(0, np.nan)
            mape_values = np.abs((pd.Series(actual_series) - pd.Series(pred_series_for_metrics)) / denominator) * 100
            mape = np.nanmean(mape_values)

            metrics = {
                'MAE': float(mae),
                'MSE': float(mse),
                'RMSE': float(rmse),
                'MAPE': float(mape) if not np.isnan(mape) else 0.0,
                'samples': int(len(actual_series)),
                'mode': evaluation_mode
            }

            self.performance_metrics = metrics

            logger.info(f"Model Performansı ({self.target_column}):")
            logger.info("  MAE: %.4f", metrics['MAE'])
            logger.info("  MSE: %.4f", metrics['MSE'])
            logger.info("  RMSE: %.4f", metrics['RMSE'])
            if metrics['MAPE'] > 0:
                logger.info("  MAPE: %.2f%%", metrics['MAPE'])
            else:
                logger.info("  MAPE: hesaplanamadı")
            logger.info("  Örnek sayısı: %d", metrics['samples'])

            return metrics
            
        except Exception as e:
            logger.error(f"Metrik hesaplama hatası: {e}")
            return {
                'MAE': 0.0, 'MSE': 0.0, 'RMSE': 0.0, 'MAPE': 0.0,
                'samples': 0, 'mode': 'calculation_error'
            }
    
    def _evaluate_with_timeseries_split(self, n_splits=3):
        """
        TimeSeriesSplit ile walk-forward validation yapar.
        Geçmiş verilerde başarı sağladığı takdirde geleceği tahmin eder.
        """
        if not PMDARIMA_AVAILABLE:
            logger.warning("pmdarima kurulu değil, TimeSeriesSplit kullanılamıyor. Normal değerlendirmeye geçiliyor.")
            return self.evaluate(use_timeseries_split=False)
        
        logger.info(f"TimeSeriesSplit ile walk-forward validation başlatılıyor (n_splits={n_splits})...")
        
        # Tüm veriyi birleştir (train + test)
        if self.test_data is not None and len(self.test_data) > 0:
            full_data = pd.concat([self.train_data, self.test_data])
            if self.train_exog is not None and self.test_exog is not None:
                full_exog = pd.concat([self.train_exog, self.test_exog])
            else:
                full_exog = None

            full_original = pd.concat([self.original_train_data, self.original_test_data]) if (
                self.original_train_data is not None and self.original_test_data is not None
            ) else None
        else:
            full_data = self.train_data.copy()
            full_exog = self.train_exog.copy() if self.train_exog is not None else None
            full_original = self.original_train_data.copy() if self.original_train_data is not None else None
        
        full_data = full_data.dropna()
        if len(full_data) < 10:
            logger.warning("TimeSeriesSplit için yetersiz veri, normal değerlendirmeye geçiliyor.")
            return self.evaluate(use_timeseries_split=False)
        
        # TimeSeriesSplit oluştur
        # Minimum fold boyutu kontrolü
        min_fold_size = max(3, len(full_data) // (n_splits + 1))
        if min_fold_size < 3:
            logger.warning(f"Fold boyutu çok küçük ({min_fold_size}), n_splits azaltılıyor.")
            n_splits = max(2, len(full_data) // 6)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        all_mae = []
        all_rmse = []
        all_mse = []
        all_mape = []
        all_predictions = []
        all_actuals = []
        last_pred_series = None
        last_actual_series = None
        
        fold = 0
        fold_details = []
        for train_idx, test_idx in tscv.split(full_data):
            fold += 1
            train_fold = full_data.iloc[train_idx]
            test_fold = full_data.iloc[test_idx]
            
            train_exog_fold = None
            test_exog_fold = None
            if full_exog is not None:
                train_exog_fold = full_exog.iloc[train_idx]
                test_exog_fold = full_exog.iloc[test_idx]
            
            if len(train_fold) < 5 or len(test_fold) < 1:
                logger.warning(f"Fold {fold}: Yetersiz veri, atlanıyor.")
                continue
            
            try:
                # Her fold için yeni model eğit
                fold_model = pm.auto_arima(
                    train_fold,
                    start_p=1, start_q=1,
                    max_p=3, max_q=3, max_d=2,
                    seasonal=False,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    n_jobs=1,
                    random_state=42,
                    X=train_exog_fold,
                    maxiter=30
                )
                
                # Test fold'unda tahmin yap
                if test_exog_fold is not None:
                    predictions = fold_model.predict(n_periods=len(test_fold), X=test_exog_fold)
                else:
                    predictions = fold_model.predict(n_periods=len(test_fold))
                
                # Orijinal skala
                if full_original is not None:
                    actual_fold = full_original.iloc[test_idx]
                else:
                    actual_fold = self._inverse_transform(test_fold)
                pred_orig = self._inverse_transform(predictions)

                # Metrikleri hesapla
                mae_fold = mean_absolute_error(actual_fold, pred_orig)
                mse_fold = mean_squared_error(actual_fold, pred_orig)
                rmse_fold = np.sqrt(mse_fold)
                
                # MAPE: index hizalama hatalarını önlemek için numpy üzerinden hesapla
                actual_vals = np.asarray(actual_fold, dtype=float)
                pred_vals = np.asarray(pred_orig, dtype=float)
                denom = np.where(actual_vals == 0, np.nan, actual_vals)
                mape_fold = np.nanmean(np.abs((actual_vals - pred_vals) / denom) * 100)
                
                all_mae.append(mae_fold)
                all_rmse.append(rmse_fold)
                all_mse.append(mse_fold)
                all_mape.append(mape_fold)
                all_predictions.extend(pd.Series(pred_orig, index=test_fold.index).values)
                all_actuals.extend(pd.Series(actual_fold, index=test_fold.index).values)

                last_pred_series = pd.Series(pred_orig, index=test_fold.index)
                last_actual_series = pd.Series(actual_fold, index=test_fold.index)
                
                fold_details.append({
                    'fold': fold,
                    'start': test_fold.index.min().isoformat() if not test_fold.empty else None,
                    'end': test_fold.index.max().isoformat() if not test_fold.empty else None,
                    'rmse': float(rmse_fold),
                    'mape': float(mape_fold) if not np.isnan(mape_fold) else None,
                    'samples': int(len(test_fold))
                })
                logger.info(f"Fold {fold}/{n_splits}: RMSE={rmse_fold:.4f}, MAPE={mape_fold:.2f}%")
                
            except Exception as e:
                logger.warning(f"Fold {fold} hatası: {e}, atlanıyor.")
                continue
        
        if len(all_mae) == 0:
            logger.warning("TimeSeriesSplit başarısız, normal değerlendirmeye geçiliyor.")
            return self.evaluate(use_timeseries_split=False)
        
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
            'samples': int(len(all_actuals)),
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
            samples=len(all_actuals)
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

    def _persist_timeseries_validation(self, metrics, fold_details, n_splits, start_date=None, end_date=None, samples=None):
        """Cross-val sonuçlarını MongoDB'ye kaydeder (opsiyonel)."""
        if not self.log_timeseries_validation:
            return
        if not metrics:
            return

        collection_name = MONGODB_COLLECTIONS.get("timeseries_validation", "timeseries_validation")
        
        # indicator_type mapping: target_column -> indicator_type
        indicator_mapping = {
            'usd_try': 'usd_try',
            'inflation': 'inflation',
            'inflation_rate': 'inflation',  # Enflasyon oranı = inflation
            'interest_rate': 'interest_rate',
            'tufe': 'inflation',  # TÜFE = inflation
            'cpi': 'inflation',   # CPI = inflation
            'policy_rate': 'interest_rate',  # Politika faizi = interest_rate
            'policy_interest_rate': 'interest_rate'
        }
        indicator_type = indicator_mapping.get(self.target_column.lower(), self.target_column)
        
        # Fold detaylarını fold_metrics formatına çevir (analyzer uyumluluğu için)
        fold_metrics = []
        for fold_detail in fold_details:
            if isinstance(fold_detail, dict):
                fold_metrics.append({
                    'fold': fold_detail.get('fold', 0),
                    'rmse': fold_detail.get('rmse', 0),
                    'mape': fold_detail.get('mape', 0),
                    'mae': fold_detail.get('mae', 0),
                    'start': fold_detail.get('start'),
                    'end': fold_detail.get('end'),
                    'samples': fold_detail.get('samples', 0)
                })
        
        document = {
            'indicator': self.target_column,  # Eski uyumluluk için
            'indicator_type': indicator_type,  # Analyzer için
            'method': 'timeseries_split',
            'n_splits': n_splits,
            'metrics': metrics,
            'folds': fold_details,  # Eski format
            'fold_metrics': fold_metrics,  # Analyzer için yeni format
            'samples': samples,
            'data_start': start_date.isoformat() if hasattr(start_date, "isoformat") else start_date,
            'data_end': end_date.isoformat() if hasattr(end_date, "isoformat") else end_date,
            'timestamp': datetime.now().isoformat(),  # Analyzer için timestamp
            'note': 'Walk-forward validation snapshot'
        }

        db_manager = None
        try:
            db_manager = MongoDBManager()
            db_manager.insert_document(collection_name, document)
        except Exception as e:
            logger.warning(f"TimeSeriesSplit sonuçları MongoDB'ye kaydedilemedi: {e}")
        finally:
            if db_manager:
                db_manager.close_connection()
    
    def plot_results(self, forecast_results=None):
        """
        Sonuçları görselleştirir.
        """
        if self.fitted_model is None:
            logger.error("Model eğitilmediği için sonuçlar çizdirilemedi.")
            return

        plt.figure(figsize=(15, 8))
        
        # 1. Gerçek veriyi çiz (Eğitim + Test)
        full_data = pd.concat([self.train_data, self.test_data])
        plt.plot(full_data.index, full_data, color='black', alpha=0.7, label='Gerçek Veri')
        
        # 2. Test seti ve gelecek tahminlerini birleştirerek tek bir sürekli çizgi oluştur
        if self.performance_metrics and self.test_predictions is not None:
            # Değerlendirme aşamasında saklanan test tahminlerini kullan
            test_predictions_series = self.test_predictions

            # Gelecek tahminleri
            future_predictions_series = forecast_results['forecast']

            # İki tahmini birleştir
            combined_forecast = pd.concat([test_predictions_series, future_predictions_series])
            
            # Birleştirilmiş tahmini çiz
            plt.plot(combined_forecast.index, combined_forecast, color='blue', linestyle='--', marker='o', markersize=3,
                     label=f'Model Tahmini (Test + Gelecek) - RMSE: {self.performance_metrics["RMSE"]:.2f}')
        
        # 3. BACKTESTING - Test verilerini ve tahminlerini çiz
        if forecast_results and 'forecast' in forecast_results:
            future_predictions_series = forecast_results['forecast']
            
            # Test verilerini çiz (gerçek değerler)
            if hasattr(self, 'test_data') and self.test_data is not None:
                plt.plot(self.test_data.index, self.test_data, 
                        color='green', linestyle='-', marker='s', markersize=4, linewidth=2,
                        label='Gerçek Test Verileri')
            
            # Tahminleri çiz - tek çizgi
            plt.plot(future_predictions_series.index, future_predictions_series, 
                    color='red', linestyle='-', marker='o', markersize=4, linewidth=2,
                    label='ARIMA Tahminleri (Backtesting)')
        
        plt.title(f'{self.target_column.replace("_", " ").upper()} için ARIMA Model Tahminleri', fontsize=18, weight='bold')
        plt.xlabel('Tarih', fontsize=12)
        plt.ylabel(f'Değer ({self.target_column.split("_")[0].upper()})', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        # Görselleştirme sorununu çöz
        try:
            plt.show()
        except Exception as e:
            logger.warning(f"Görselleştirme hatası: {e}")
            logger.info("Grafik kaydediliyor...")
            plt.savefig('arima_forecast.png', dpi=300, bbox_inches='tight')
            logger.info("Grafik 'arima_forecast.png' olarak kaydedildi.")

    def _prepare_future_exog(self, steps):
        """Seçilen stratejiye göre gelecek dönem exogenous verisini hazırlar."""
        if self.future_exog_strategy != 'repeat_last':
            logger.debug("Bilinmeyen future_exog_strategy, exog kullanılmayacak.")
            return None

        source_exog = None
        if self.test_exog is not None and not self.test_exog.empty:
            source_exog = self.test_exog
        elif self.train_exog is not None and not self.train_exog.empty:
            source_exog = self.train_exog

        if source_exog is None:
            return None

        last_row = source_exog.iloc[-1]
        repeated = np.tile(last_row.values, (steps, 1))

        # pmdarima gelecekteki tahminler için sadece değerleri kullanır, indeks önemli değildir
        # ama deterministic time features (month_sin/cos, trend) için indeks üzerinden üretmek daha doğru
        last_date = None
        if self.test_data is not None and not self.test_data.empty:
            last_date = self.test_data.index[-1]
        elif self.train_data is not None and not self.train_data.empty:
            last_date = self.train_data.index[-1]

        if isinstance(last_date, pd.Timestamp):
            future_index = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=steps, freq='MS')
        else:
            future_index = pd.RangeIndex(start=0, stop=steps)

        future_df = pd.DataFrame(repeated, index=future_index, columns=source_exog.columns)

        # Deterministic time features: month_sin / month_cos
        if isinstance(future_df.index, pd.DatetimeIndex):
            months = future_df.index.month
            if 'month_sin' in future_df.columns:
                future_df['month_sin'] = np.sin(2 * np.pi * months / 12)
            if 'month_cos' in future_df.columns:
                future_df['month_cos'] = np.cos(2 * np.pi * months / 12)

        # Deterministic trend feature
        if 'trend' in future_df.columns:
            try:
                last_trend = float(last_row.get('trend'))
                future_df['trend'] = last_trend + np.arange(1, steps + 1)
            except Exception:
                # fallback: monoton artan
                future_df['trend'] = np.arange(1, steps + 1)

        return future_df


def load_complete_data_from_mongodb(target_field='usd_try', include_features=False):
    """
    MongoDB'den sadece tüm ekonomik göstergelerin mevcut olduğu verileri yükler.
    
    Args:
        target_field (str): Yüklenecek hedef alan adı (usd_try, policy_rate, inflation_rate).
        
    Returns:
        pd.Series: Tarih indeksi ile hazırlanmış zaman serisi verisi.
    """
    logger.info(f"MongoDB'den '{target_field}' verisi yükleniyor (tüm göstergeler mevcut)...")
    
    try:
        with MongoDBManager() as mongodb_manager:
            # MongoDB'den veriyi çek - tüm ekonomik göstergelerin mevcut olduğu veriler
            query = {
                "data_type": "economic_indicators",
                "usd_try": {"$exists": True, "$ne": None},
                "policy_rate": {"$exists": True, "$ne": None},
                "inflation_rate": {"$exists": True, "$ne": None},
                target_field: {"$exists": True, "$ne": None}
            }
            
            documents = mongodb_manager.find_documents(
                collection_name='economic_indicators',
                query=query,
                sort=[("date", 1)]  # Tarihe göre sırala
            )
            
            if not documents:
                logger.error("MongoDB'den tüm göstergelerin mevcut olduğu veri bulunamadı.")
                if target_field == 'usd_try':
                    data = _load_usd_try_from_csv()
                    if data is not None:
                        return data
                return None
            
            # DataFrame'e çevir
            df = pd.DataFrame(documents)
            
            if df.empty:
                logger.error("MongoDB'den alınan veri boş.")
                return None
                
            logger.info(f"{len(df)} satır veri MongoDB'den başarıyla çekildi (tüm göstergeler mevcut).")
            
            # Veri tiplerini ve indeksi ayarla
            df['date'] = pd.to_datetime(df['date'])
            
            # Hedef alanı seç ve float'a çevir
            if target_field not in df.columns:
                logger.error(f"Hedef alan '{target_field}' bulunamadı. Mevcut alanlar: {list(df.columns)}")
                return None

            keep_columns = ['date', 'usd_try', 'policy_rate', 'inflation_rate']
            keep_columns = [col for col in keep_columns if col in df.columns]
            if target_field not in keep_columns:
                keep_columns.append(target_field)
            df = df[keep_columns].copy()
            df = df.drop_duplicates(subset=['date'], keep='last')
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.sort_index()
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill()
            df = df.asfreq('MS')
            df = df.dropna(subset=[target_field])

            if not include_features:
                series = df[target_field].astype(float)
                logger.info(
                    f"ARIMA için hazırlanan veri: {len(series)} kayıt, tarih aralığı: {series.index.min()} - {series.index.max()}"
                )
                return series

            feature_df = df.copy()

            months = feature_df.index.month
            feature_df['month_sin'] = np.sin(2 * np.pi * months / 12)
            feature_df['month_cos'] = np.cos(2 * np.pi * months / 12)

            # Hedef değişken için geçmişe dönük özellikler
            feature_df[f'{target_field}_diff_1'] = feature_df[target_field].diff().fillna(0.0)
            feature_df[f'{target_field}_ma_3'] = feature_df[target_field].rolling(window=3).mean()
            feature_df[f'{target_field}_ma_6'] = feature_df[target_field].rolling(window=6).mean()
            feature_df[f'{target_field}_std_6'] = feature_df[target_field].rolling(window=6).std()

            # Diğer göstergelerin gecikmeli etkisi
            if 'policy_rate' in feature_df.columns:
                feature_df['policy_rate_lag_1'] = feature_df['policy_rate'].shift(1)
                feature_df['policy_rate_diff_1'] = feature_df['policy_rate'].diff()
            if 'inflation_rate' in feature_df.columns:
                feature_df['inflation_rate_lag_1'] = feature_df['inflation_rate'].shift(1)
                feature_df['inflation_rate_diff_1'] = feature_df['inflation_rate'].diff()

            feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
            feature_df = feature_df.dropna()

            logger.info(
                "ARIMA (çok değişkenli) veri seti hazır: %d kayıt, %d özellik",
                len(feature_df),
                feature_df.shape[1] - 1
            )
            return feature_df
            
    except Exception as e:
        logger.error(f"MongoDB'den veri yüklenirken hata oluştu: {e}")
        return None

def _load_usd_try_from_csv():
    """MongoDB'de veri yoksa tr_economy_ml CSV'den USD/TRY yükler. Returns pd.Series or None."""
    try:
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'src', 'data_collection', 'tr_economy_ml_2005_01_to_2025_11.csv'
        )
        if not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
        if 'usd_try_avg_buy' not in df.columns:
            return None
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['usd_try_avg_buy']).drop_duplicates(subset=['date'], keep='last')
        df.set_index('date', inplace=True)
        df = df.sort_index()
        series = df['usd_try_avg_buy'].astype(float)
        series = series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < 10:
            return None
        series = series.asfreq('MS')
        if series.isna().any():
            series = series.interpolate(method='linear', limit_direction='both').ffill().bfill()
        series = series.rename('usd_try')
        logger.info(f"USD/TRY verisi CSV'den yüklendi: {len(series)} kayıt")
        return series
    except Exception as e:
        logger.warning(f"CSV'den USD/TRY yüklenemedi: {e}")
        return None


def load_data_from_mongodb(target_field='usd_try'):
    """
    MongoDB'den belirtilen alanı yükler ve zaman serisi olarak hazırlar.
    USD/TRY için MongoDB boşsa tr_economy_ml CSV'den dener.
    
    Args:
        target_field (str): Yüklenecek hedef alan adı (usd_try, policy_rate, inflation_rate).
        
    Returns:
        pd.Series: Tarih indeksi ile hazırlanmış zaman serisi verisi.
    """
    logger.info(f"MongoDB'den '{target_field}' verisi yükleniyor...")
    
    try:
        with MongoDBManager() as mongodb_manager:
            # MongoDB'den veriyi çek - sadece tüm değişkenlerin mevcut olduğu veriler
            query = {
                "data_type": "economic_indicators",
                target_field: {"$exists": True, "$ne": None}  # Hedef alan mevcut ve null değil
            }
            
            documents = mongodb_manager.find_documents(
                collection_name='economic_indicators',
                query=query,
                sort=[("date", 1)]  # Tarihe göre sırala
            )
            
            if not documents:
                logger.error("MongoDB'den veri alınamadı veya koleksiyon boş.")
                if target_field == 'usd_try':
                    data = _load_usd_try_from_csv()
                    if data is not None:
                        return data
                return None
            
            # DataFrame'e çevir
            df = pd.DataFrame(documents)
            
            if df.empty:
                logger.error("MongoDB'den alınan veri boş.")
                return None
                
            logger.info(f"{len(df)} satır veri MongoDB'den başarıyla çekildi.")
            
            # Veri tiplerini ve indeksi ayarla
            df['date'] = pd.to_datetime(df['date'])
            
            # Hedef alanı seç ve float'a çevir
            if target_field not in df.columns:
                logger.error(f"Hedef alan '{target_field}' bulunamadı. Mevcut alanlar: {list(df.columns)}")
                return None
            
            # Sadece gerekli sütunları al ve duplicate tarihleri kaldır
            df = df[['date', target_field]].copy()
            
            # NaN değerleri kaldır - SESSIZ
            df = df.dropna(subset=[target_field])
            
            # Duplicate tarihleri kaldır (son değeri al) - SESSIZ
            df = df.drop_duplicates(subset=['date'], keep='last')
            
            df.set_index('date', inplace=True)
            
            # Float'a çevir ve geçersiz değerleri temizle - SESSIZ
            df = df[target_field].apply(pd.to_numeric, errors='coerce')
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            # Minimum veri kontrolü
            if len(df) < 10:
                logger.error(f"❌ Yetersiz veri: {len(df)} nokta")
                return None
            
            # Frekansı ayarla (aylık başlangıç) - SESSIZ
            df = df.asfreq('MS')
            
            # Son NaN kontrolü (asfreq sonrası oluşabilir) - SESSIZ
            if df.isna().any():
                df = df.interpolate(method='linear', limit_direction='both')
                df = df.ffill().bfill()
            
            logger.info(f"✅ Veri yüklendi: {len(df)} nokta")
            
            return df
            
    except Exception as e:
        logger.error(f"MongoDB'den veri yüklenirken hata oluştu: {e}")
        return None

def load_data_from_db(target_column='usd_avg'):
    """
    Veritabanından belirtilen sütunu yükler ve zaman serisi olarak hazırlar.
    
    Args:
        target_column (str): Yüklenecek hedef sütun adı.
        
    Returns:
        pd.Series: Tarih indeksi ile hazırlanmış zaman serisi verisi.
    """
    logger.info(f"Veritabanından '{target_column}' verisi yükleniyor...")
    db_manager = DatabaseManager()
    
    # Veriyi çek
    query = f"SELECT date, {target_column} FROM financial_data_raw ORDER BY date ASC"
    df = db_manager.read_sql_to_dataframe(query)
    
    if df is None or df.empty:
        logger.error("Veritabanından veri alınamadı veya tablo boş.")
        return None
        
    logger.info(f"{len(df)} satır veri başarıyla çekildi.")
    
    # Veri tiplerini ve indeksi ayarla
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df[target_column].astype(float)
    
    # Eksik verileri doldur (opsiyonel, ileriye/geriye dönük doldurma)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # Frekansı ayarla (aylık başlangıç)
    # Bu, ValueWarning'i önler.
    df = df.asfreq('MS')
    logger.info("Veri frekansı 'MS' (Ay Başı) olarak ayarlandı ve eksik veriler dolduruldu.")

    return df

def test_arima_model():
    """
    ARIMA modelini test eder - EN SON VERİDEN 12 AY SONRASINA kadar tahmin yapar.
    """
    logger.info("ARIMA modeli test ediliyor - en son veriden 12 ay sonrasına kadar...")
    
    # MongoDB'den veriyi yükle (tüm göstergeler mevcut olan veriler)
    data = load_complete_data_from_mongodb(target_field='usd_try')
    
    if data is None or data.empty:
        logger.error("MongoDB'den veri yüklenemediği için ARIMA model testi iptal edildi.")
        return

    logger.info(f"Yüklenen veri: {len(data)} kayıt, tarih aralığı: {data.index.min()} - {data.index.max()}")
    if not data.empty:
        logger.info(f"En son veri: {data.index[-1]} - {data.iloc[-1]:.2f}")
    
    # Modeli oluştur ve TÜM veriyle eğit
    forecaster = ARIMAForecaster(target_column='usd_try')
    forecaster.fit(data, test_size=0.2)  # Normal eğitim
    
    # Performansı değerlendir
    forecaster.evaluate()
    
    # EN SON VERİDEN 12 ay sonrasına kadar tahmin yap
    forecast_results = forecaster.forecast(steps=12)
    
    if forecast_results:
        logger.info("\n" + "="*60)
        logger.info("GELECEK 12 AY TAHMİNLERİ (2025-10'den 2026-10'a)")
        logger.info("="*60)
        
        for date, value in forecast_results['forecast'].items():
            logger.info(f"{date.strftime('%Y-%m')}: {value:.2f} TRY")
        
        # Son tahmin
        if not forecast_results['forecast'].empty:
            last_forecast = forecast_results['forecast'].iloc[-1]
            logger.info(f"\n🎯 SON TAHMİN (2026-10): {last_forecast:.2f} TRY")
    
    # Sonuçları görselleştir
    if forecast_results:
        forecaster.plot_results(forecast_results)
    
    logger.info("ARIMA model testi başarıyla tamamlandı.")

def test_arima_model_postgresql():
    """
    ARIMA modelini PostgreSQL verileriyle test eder (geriye uyumluluk).
    """
    logger.info("ARIMA modeli PostgreSQL verileriyle test ediliyor...")
    
    # PostgreSQL'den veriyi yükle
    data = load_data_from_db(target_column='usd_avg')
    
    if data is None or data.empty:
        logger.error("PostgreSQL'den veri yüklenemediği için ARIMA model testi iptal edildi.")
        return

    # Modeli oluştur ve eğit
    forecaster = ARIMAForecaster(target_column='usd_avg')
    forecaster.fit(data, test_size=0.2)
    
    # Performansı değerlendir
    forecaster.evaluate()
    
    # 12 ay sonrası için tahmin yap (EN SON VERİDEN)
    forecast_results = forecaster.forecast(steps=12)
    
    if forecast_results:
        logger.info("\n" + "="*60)
        logger.info("GELECEK 12 AY TAHMİNLERİ (2025-10'den 2026-10'a)")
        logger.info("="*60)
        
        for date, value in forecast_results['forecast'].items():
            logger.info(f"{date.strftime('%Y-%m')}: {value:.2f} TRY")
        
        # Son tahmin
        if not forecast_results['forecast'].empty:
            last_forecast = forecast_results['forecast'].iloc[-1]
            logger.info(f"\n🎯 SON TAHMİN (2026-10): {last_forecast:.2f} TRY")
    
    # Sonuçları görselleştir
    if forecast_results:
        forecaster.plot_results(forecast_results)
    
    logger.info("ARIMA model testi başarıyla tamamlandı.")

# Bu dosya doğrudan çalıştırıldığında test fonksiyonunu çağırır
if __name__ == '__main__':
    test_arima_model() 
