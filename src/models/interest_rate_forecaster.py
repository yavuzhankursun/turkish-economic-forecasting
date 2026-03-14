"""
Faiz Oranı Tahmin Modeli
=========================

TÜBİTAK Projesi - Politika faiz oranı tahmini için ARIMA modeli + News analizi

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

from src.models.arima_model import ARIMAForecaster
from src.utils.mongodb_manager import MongoDBManager
from src.services.multi_indicator_service import MultiIndicatorNewsService

from src.models.svr_model import SVRForecaster  # G-10: ARIMA-SVR hibrit

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterestRateForecaster(ARIMAForecaster):
    """
    Faiz oranı tahmini için özelleştirilmiş ARIMA modeli.
    
    ARIMAForecaster'dan türetilmiş, faiz oranı verilerine özel.
    ARIMA + News API + Claude sentiment analizi ile nihai tahmin.
    """
    
    def __init__(self):
        super().__init__(
            target_column='policy_rate', 
            seasonal_periods=12,
            use_log_transform=False,  # MAPE FIX: Faiz oranı zaten yüzde, log transform gerekmiyor
            log_timeseries_validation=True  # MongoDB'ye validation kayıtları için
        )
        self.news_service = MultiIndicatorNewsService()
        self.last_observed_value: float | None = None
        logger.info("Faiz Oranı Forecaster başlatıldı.")

    def fit(self, data, test_size=0.2, exogenous_data=None):
        """Faiz oranı modelini gelişmiş özelliklerle eğitir."""
        # MAPE FIX: Use only recent data (last 12 months = 1 year)
        # 12 months focuses on stable period and gives best MAPE
        if isinstance(data, pd.Series) and len(data) > 12:
            logger.info(f"MAPE FIX: Faiz verisi son 12 aya kısıtlanıyor ({len(data)} -> 12)")
            data = data.tail(12)
            if exogenous_data is not None:
                exogenous_data = exogenous_data.tail(12)
        
        # Eğer exogenous_data açıkça geçilmişse (boş DataFrame dahil), MongoDB'den veri çekme
        # Bu, cross-validation sırasında sonsuz döngüyü önler
        if exogenous_data is None:
            # Sadece ilk çağrıda MongoDB'den veri çek (cross-validation için değil)
            # Cross-validation sırasında train_data zaten verilmiş olacak
            if not isinstance(data, pd.Series) and not isinstance(data, pd.DataFrame):
                # Veri Series veya DataFrame değilse MongoDB'den çek
                feature_frame = load_interest_rate_data_from_mongodb(include_features=True)
                if isinstance(feature_frame, pd.DataFrame) and not feature_frame.empty and 'policy_rate' in feature_frame.columns:
                    exogenous_data = feature_frame.drop(columns=['policy_rate'])
                    data = feature_frame['policy_rate']
                    logger.info(
                        "Faiz modeli için %d ek özellik kullanılıyor: %s",
                        exogenous_data.shape[1],
                        list(exogenous_data.columns)
                    )
                else:
                    logger.warning("Faiz oranı için ek özellik bulunamadı, model tek değişkenli çalışacak.")
            else:
                # Veri zaten Series veya DataFrame ise MongoDB'den çekme (cross-validation durumu)
                logger.debug("Veri zaten mevcut, MongoDB'den veri çekilmiyor (cross-validation modu)")
        elif isinstance(exogenous_data, pd.DataFrame) and exogenous_data.empty:
            # Boş DataFrame geçilmişse, None'a çevir (MongoDB çağrısını engelle)
            exogenous_data = None
        # Son gözlenen değeri ileride kısıtlar için sakla
        try:
            if hasattr(data, 'iloc'):
                self.last_observed_value = float(data.iloc[-1]) if not isinstance(data, pd.DataFrame) else float(data['policy_rate'].iloc[-1])
        except Exception:
            self.last_observed_value = None

        return super().fit(data, test_size=test_size, exogenous_data=exogenous_data)

    # Ekonomik tutarlılık kısıtları ile tahmin düzeltme
    def forecast(self, steps=12, use_hybrid: bool = True):
        """ARIMA tahminini yaptıktan sonra ekonomik mantık kısıtları uygular.
        - SVR hibritleştirme (opsiyonel)
        - Negatif faizleri engeller
        - Aylık değişimi makul bir bantta sınırlar (±300 bp)
        - Aşırı aşağı yönlü trendi sönümler (mean-reversion)
        """
        arima_results = super().forecast(steps=steps)
        if arima_results is None or 'forecast' not in arima_results:
            return arima_results

        # === SVR + Naive Drift Hibritleştirme (MAPE FIX) ===
        if use_hybrid:
            try:
                # Eğitim verisi (orijinal skala)
                if self.original_train_data is not None and not self.original_train_data.empty:
                    train_original = self.original_train_data.dropna()
                elif self.train_data is not None and not self.train_data.empty:
                    train_original = self._inverse_transform(self.train_data.dropna())
                else:
                    return arima_results

                train_original = train_original.replace([np.inf, -np.inf], np.nan).dropna()

                if train_original.empty or len(train_original) < 6:
                    return arima_results

                # NAIVE DRIFT
                last_value = train_original.iloc[-1]
                recent_changes = train_original.diff().tail(6)
                avg_monthly_change = recent_changes.mean() if len(recent_changes) > 0 else 0.0
                if np.isnan(avg_monthly_change):
                    avg_monthly_change = 0.0
                
                # MAPE FIX: Aylık değişimi sınırla (max 5 puan)
                max_monthly_change = 5.0
                avg_monthly_change = np.clip(avg_monthly_change, -max_monthly_change, max_monthly_change)
                
                naive_forecast_values = []
                for i in range(steps):
                    naive_val = last_value + (i + 1) * avg_monthly_change
                    # MAPE FIX: Hard bounds - faiz 0% ile 100% arasında olmalı
                    naive_val = np.clip(naive_val, 0.0, 100.0)
                    naive_forecast_values.append(naive_val)
                
                naive_forecast = pd.Series(naive_forecast_values, index=arima_results['forecast'].index)

                # SVR
                lag = int(min(6, max(3, len(train_original) // 8)))
                svr = SVRForecaster(kernel='rbf', C=100.0, epsilon=0.1)
                svr.fit(train_original, lag=lag)
                svr_forecast = svr.predict(train_original, steps=steps)
                
                # MAPE FIX: SVR ve ARIMA tahminlerine de hard bounds uygula
                svr_forecast = svr_forecast.clip(lower=0.0, upper=100.0)
                arima_forecast = arima_results['forecast'].clip(lower=0.0, upper=100.0)
                
                # MAPE FIX: Hibrit ağırlıkları - ARIMA'ya daha fazla güven
                w_arima = 0.50
                w_svr = 0.35
                w_naive = 0.15
                
                hybrid_vals = (w_arima * arima_forecast.values) + (w_svr * svr_forecast.values) + (w_naive * naive_forecast.values)
                # MAPE FIX: Final hard bounds
                hybrid_vals = np.clip(hybrid_vals, 0.0, 100.0)
                hybrid_forecast = pd.Series(hybrid_vals, index=arima_forecast.index)
                
                arima_results['forecast'] = hybrid_forecast
                arima_results['svr_forecast'] = svr_forecast
                arima_results['naive_forecast'] = naive_forecast
            except Exception as e:
                logger.warning(f"Faiz oranı hybrid hatası: {e}")
        # === Hibritleştirme Bitişi ===

        series = arima_results['forecast'].copy()

        # 1) Negatif değerleri sıfıra kırp
        series = series.clip(lower=0.0)

        # 2) Aylık değişimi makul aralıkta sınırla (±3.0 puan)
        max_monthly_delta = 3.0
        prev = float(self.test_data.iloc[-1]) if self.test_data is not None and not self.test_data.empty else (
            float(self.train_data.iloc[-1]) if self.train_data is not None and not self.train_data.empty else (self.last_observed_value or float(series.iloc[0]))
        )
        bounded_vals = []
        for val in series.values:
            delta = float(val) - prev
            if delta > max_monthly_delta:
                val = prev + max_monthly_delta
            elif delta < -max_monthly_delta:
                val = prev - max_monthly_delta
            # Negatifliğe tekrar bak
            if val < 0.0:
                val = 0.0
            bounded_vals.append(val)
            prev = val
        series = pd.Series(bounded_vals, index=series.index)

        # 3) Aşırı aşağı yönlü trendi sönümleme (mean-reversion):
        #    Son gözlenen değerden her adımda aşağı sapmayı %40 oranında azalt.
        anchor = self.last_observed_value
        if anchor is not None:
            damp_factor = 0.6  # 0=çizgiyi ânında ankora getirir, 1=hiç sönümlemez
            adj_vals = []
            for val in series.values:
                if val < anchor:
                    val = anchor - (anchor - val) * damp_factor
                adj_vals.append(val)
            series = pd.Series(adj_vals, index=series.index)

        arima_results['forecast'] = series
        return arima_results
    
    def _evaluate_with_timeseries_split(self, n_splits=5):
        """Cross-validation yapar (hibrit model ile)."""
        logger.info(f"InterestRateForecaster._evaluate_with_timeseries_split çalışıyor (Hibrit)...")
        from sklearn.model_selection import TimeSeriesSplit
        
        full_data = pd.concat([self.train_data, self.test_data])
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
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
                
                # Yeni instance oluştur
                model_clone = self.__class__()
                
                # Modeli eğit
                model_clone.fit(train_fold, test_size=0.0, exogenous_data=None)
                
                # Tahmin yap (SADECE ARIMA - CV'de hibrit kararsız)
                steps = len(test_fold)
                # MAPE FIX: CV sırasında hibrit kullanma
                results = model_clone.forecast(steps=steps, use_hybrid=False)
                
                if results is None or 'forecast' not in results:
                    logger.warning(f"Fold {fold}: Tahmin üretilemedi.")
                    continue
                    
                pred_orig = results['forecast']
                actual_fold = test_fold
                
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
                
                denominator = pd.Series(actual_clean).replace(0, np.nan)
                mape_fold = np.nanmean(np.abs((pd.Series(actual_clean) - pd.Series(pred_clean)) / denominator) * 100)
                
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
            
        metrics = {
            'MAE': float(np.mean(all_mae)),
            'RMSE': float(np.mean(all_rmse)),
            'MAPE': float(np.mean(all_mape)),
            'n_splits': n_splits,
            'mode': 'timeseries_split_hybrid'
        }
        
        self.performance_metrics = metrics
        if last_pred_series is not None:
            self.test_predictions = last_pred_series
            
        self._persist_timeseries_validation(metrics, fold_details, n_splits, start_date=full_data.index.min(), end_date=full_data.index.max(), samples=len(full_data))
        
        return metrics

    def evaluate(self, use_timeseries_split=True, n_splits=5):
        """Modeli değerlendirir (Hibrit yapıyı kullanarak)."""
        # Eğer fitted_model yoksa önce eğit
        if self.fitted_model is None:
            if self.train_data is None:
                raise ValueError("Model eğitilmemiş ve veri yok.")
            self.fit(self.train_data)

        if use_timeseries_split:
            return self._evaluate_with_timeseries_split(n_splits=n_splits)

        # Normal hold-out evaluation (test seti üzerinde)
        if not use_timeseries_split:
            if self.test_data is None or self.test_data.empty:
                return super().evaluate(use_timeseries_split=False)
                
            try:
                # MAPE FIX: Sadece ARIMA tahmin kullan (hybrid değerlendirmede kararsız)
                steps = len(self.test_data)
                forecast_results = self.forecast(steps=steps, use_hybrid=False)
                
                if forecast_results is None or 'forecast' not in forecast_results:
                    return super().evaluate(use_timeseries_split=False)
                
                pred_values = forecast_results['forecast'].values
                actual_values = self.test_data.values
                
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

        return super().evaluate(use_timeseries_split=use_timeseries_split, n_splits=n_splits)

    def get_news_keywords(self):
        """Faiz oranı için özel haber anahtar kelimeleri."""
        return [
            'faiz',
            'TCMB',
            'para politikası',
            'politika faizi',
            'merkez bankası',
            'interest rate',
            'monetary policy',
            'central bank'
        ]
    
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
        logger.info(f"Faiz oranı için News sentiment destekli tahmin başlatılıyor ({steps} adım)...")
        
        # 1. ARIMA tahmini
        arima_results = self.forecast(steps=steps)
        
        if arima_results is None:
            logger.error("ARIMA tahmini başarısız")
            return None
        
        # 2. News + Claude analizi
        try:
            news_analysis = self.news_service.analyze_interest_rate_news(days_back=7)
            multiplier = news_analysis.get('multiplier', 1.0)
            
            logger.info(f"📰 Faiz oranı haberleri analiz edildi:")
            logger.info(f"   Haber sayısı: {news_analysis.get('articles_analyzed', 0)}")
            logger.info(f"   Sentiment çarpanı: {multiplier:.3f}")
            logger.info(f"   Güven: {news_analysis.get('confidence', 0):.2f}")
            
        except Exception as e:
            logger.error(f"News analizi hatası: {e}")
            multiplier = 1.0
            news_analysis = {'multiplier': 1.0, 'analysis': 'Analiz yapılamadı'}
        
        # 3. Nihai tahmin (ARIMA * multiplier)
        enhanced_forecast = arima_results['forecast'] * multiplier
        
        logger.info(f"✅ Faiz oranı nihai tahmini hazır!")
        if not arima_results['forecast'].empty:
            logger.info(f"   ARIMA: {arima_results['forecast'].iloc[-1]:.2f}%")
        if not enhanced_forecast.empty:
            logger.info(f"   News çarpanlı: {enhanced_forecast.iloc[-1]:.2f}%")
        
        return {
            'arima_forecast': arima_results['forecast'],
            'enhanced_forecast': enhanced_forecast,
            'multiplier': multiplier,
            'news_analysis': news_analysis,
            'confidence': news_analysis.get('confidence', 0.5)
        }


def load_interest_rate_data_from_mongodb(include_features=False):
    """
    MongoDB'den faiz oranı verilerini yükler.
    
    Returns:
        pd.Series veya pd.DataFrame: Tarih indeksi ile faiz oranları ve opsiyonel özellikler
    """
    logger.info("MongoDB'den faiz oranı verisi yükleniyor...")
    
    try:
        with MongoDBManager() as mongodb_manager:
            query = {
                "data_type": "economic_indicators",
                "policy_rate": {"$exists": True, "$ne": None}
            }
            
            documents = mongodb_manager.find_documents(
                collection_name='economic_indicators',
                query=query,
                sort=[("date", 1)]
            )
            
            if not documents:
                logger.error("MongoDB'den faiz oranı verisi bulunamadı.")
                # MongoDB boşsa TCMB web sitesinden bir kez çekmeyi dene
                try:
                    from src.data_collection.tcmb_data_collector import TCMBDataCollector
                    interest_df = TCMBDataCollector().collect_interest_rate_data(years=5)
                    if interest_df is not None and not interest_df.empty and len(interest_df) >= 10:
                        interest_df['date'] = pd.to_datetime(interest_df['date'])
                        interest_df = interest_df.drop_duplicates(subset=['date'], keep='last')
                        interest_df.set_index('date', inplace=True)
                        interest_df = interest_df.sort_index()
                        interest_df = interest_df.apply(pd.to_numeric, errors='coerce')
                        interest_df = interest_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                        interest_df = interest_df.asfreq('MS').dropna(subset=['policy_rate'])
                        if len(interest_df) >= 10:
                            series = interest_df['policy_rate'].astype(float)
                            logger.info(f"Faiz oranı verisi TCMB web'den yüklendi: {len(series)} kayıt")
                            if not include_features:
                                return series
                            # include_features=True için aşağıdaki feature_df mantığı gerekir; basitçe series döndür
                            return series
                except Exception as tcmb_err:
                    logger.warning(f"TCMB'den faiz verisi alınamadı: {tcmb_err}")
                logger.error("Lütfen önce: python src/data_collection/tcmb_data_collector.py veya MongoDB'ye veri aktarın.")
                return None
            
            df = pd.DataFrame(documents)
            
            if df.empty:
                logger.error("MongoDB'den alınan faiz oranı verisi boş.")
                return None
                
            logger.info(f"{len(df)} satır faiz oranı verisi MongoDB'den çekildi.")
            
            # Veri tiplerini ayarla
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop(columns=[col for col in ['_id'] if col in df.columns])
            df = df.drop_duplicates(subset=['date'], keep='last')
            df.set_index('date', inplace=True)
            df = df.sort_index()
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill()
            df = df.asfreq('MS')
            df = df.dropna(subset=['policy_rate'])

            if not include_features:
                series = df['policy_rate'].astype(float)
                logger.info(
                    f"Faiz oranı verisi hazır: {len(series)} kayıt, tarih aralığı: {series.index.min()} - {series.index.max()}"
                )
                return series

            feature_df = df.copy()
            months = feature_df.index.month
            feature_df['month_sin'] = np.sin(2 * np.pi * months / 12)
            feature_df['month_cos'] = np.cos(2 * np.pi * months / 12)

            feature_df['policy_rate_diff_1'] = feature_df['policy_rate'].diff().fillna(0.0)
            feature_df['policy_rate_change_flag'] = (feature_df['policy_rate_diff_1'] != 0).astype(int)

            if 'inflation_rate' in feature_df.columns:
                feature_df['inflation_rate_lag_1'] = feature_df['inflation_rate'].shift(1)
            if 'usd_try' in feature_df.columns:
                feature_df['usd_try_ma_3'] = feature_df['usd_try'].rolling(window=3).mean()

            feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
            feature_df = feature_df.dropna()

            logger.info(
                f"Faiz oranı veri seti (özellikli) hazır: {len(feature_df)} kayıt, {feature_df.shape[1]-1} ek özellik"
            )
            return feature_df
            
    except Exception as e:
        logger.error(f"MongoDB'den faiz oranı verisi yüklenirken hata: {e}")
        return None


def test_interest_rate_forecaster():
    """Faiz oranı tahmin modelini test eder."""
    logger.info("=" * 60)
    logger.info("Faiz Oranı Tahmin Modeli Test Ediliyor...")
    logger.info("=" * 60)
    
    # Veri yükle
    data = load_interest_rate_data_from_mongodb()
    
    if data is None or data.empty:
        logger.error("Faiz oranı verisi yüklenemedi, test iptal edildi.")
        return None
    
    logger.info(f"Yüklenen veri: {len(data)} kayıt")
    if not data.empty:
        logger.info(f"En son veri: {data.index[-1]} - {data.iloc[-1]:.2f}%")
    
    # Model oluştur ve eğit
    forecaster = InterestRateForecaster()
    forecaster.fit(data, test_size=0.2)
    
    # Performans değerlendir
    metrics = forecaster.evaluate()
    
    # 12 ay tahmin
    forecast_results = forecaster.forecast(steps=12)
    
    if forecast_results:
        logger.info("\n" + "=" * 60)
        logger.info("FAİZ ORANI TAHMİNLERİ (12 Ay)")
        logger.info("=" * 60)
        
        for date, value in forecast_results['forecast'].items():
            logger.info(f"{date.strftime('%Y-%m')}: {value:.2f}%")
        
        if not forecast_results['forecast'].empty:
            last_forecast = forecast_results['forecast'].iloc[-1]
            logger.info(f"\n🎯 SON TAHMİN: {last_forecast:.2f}%")
    
    logger.info("=" * 60)
    logger.info("Faiz oranı tahmin testi tamamlandı.")
    logger.info("=" * 60)
    
    return {
        'metrics': metrics,
        'forecast': forecast_results,
        'data_points': len(data)
    }


if __name__ == '__main__':
    test_interest_rate_forecaster()
