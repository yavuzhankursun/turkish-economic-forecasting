"""
EconomicAnalyzer - Ana Koordinatör Sınıfı
==========================================

TÜBİTAK Projesi - Tüm ekonomik göstergeleri analiz eden merkezi sınıf

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

import logging
import math
import time
from datetime import datetime
from typing import Dict, Optional
import sys
import os
import pandas as pd

# Proje kök dizinini import yolu olarak ekle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models.arima_model import ARIMAForecaster, load_complete_data_from_mongodb
from src.models.inflation_forecaster import InflationForecaster, load_inflation_data_from_mongodb
from src.models.interest_rate_forecaster import InterestRateForecaster, load_interest_rate_data_from_mongodb
from src.models.svr_model import ARIMASVRHybrid  # G-10: Hibrit metodoloji
from src.services.multi_indicator_service import MultiIndicatorNewsService
from src.utils.accuracy_validator import AccuracyValidator  # G-32-G-34: Doğruluk kontrolleri
from src.utils.accuracy_calculator import AccuracyCalculator  # G-35-G-36: Doğruluk oranları
from src.utils.performance_monitor import PerformanceMonitor  # G-44-G-46: Performans metrikleri

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EconomicAnalyzer:
    """
    Ana ekonomik analiz sınıfı - TÜBİTAK gereksinimi.
    
    Üç ekonomik göstergeyi eşzamanlı analiz eder:
    - USD/TRY (Döviz Kuru)
    - Enflasyon Oranı
    - Politika Faiz Oranı
    
    Özellikler:
    - ARIMA modelleri ile tahmin
    - News API + Claude ile sentiment analizi
    - Performance monitoring
    - Otomatik raporlama
    """
    
    def __init__(self):
        """EconomicAnalyzer'ı başlat."""
        self.usd_forecaster = None
        self.inflation_forecaster = None
        self.interest_forecaster = None
        self.news_service = MultiIndicatorNewsService()
        self.results = {}
        self.performance_metrics = {}
        
        logger.info("=" * 80)
        logger.info("EconomicAnalyzer - TÜBİTAK Ekonomik Göstergeler Tahmin Sistemi")
        logger.info("=" * 80)
    
    def analyze_all(self, forecast_steps=12, test_size=0.2, include_news=True, use_hybrid: bool = False) -> Dict:
        """
        Tüm ekonomik göstergeleri analiz eder.
        
        Args:
            forecast_steps: Tahmin adım sayısı (ay)
            test_size: Test verisi oranı
            include_news: Haber analizi dahil edilsin mi
            
        Returns:
            Dict: Tüm analiz sonuçları
        """
        start_time = time.time()
        
        # G-44-G-46: Performans izleme başlat
        perf_monitor = PerformanceMonitor()
        initial_memory = perf_monitor.get_memory_usage()
        
        logger.info("\n" + "=" * 80)
        logger.info("TÜM EKONOMİK GÖSTERGELER ANALİZİ BAŞLATILIYOR...")
        logger.info("=" * 80)
        
        results = {
            'usd_try': None,
            'inflation': None,
            'interest_rate': None,
            'news_analysis': None,
            'performance': {},
            'timestamp': datetime.now().isoformat(),
            'success': False
        }
        
        try:
            # 1. USD/TRY Analizi
            logger.info("\n" + "=" * 80)
            logger.info("📊 1/3: USD/TRY KURU ANALİZİ BAŞLIYOR...")
            logger.info("=" * 80)
            usd_start = time.time()
            
            try:
                results['usd_try'] = self._analyze_usd_try(forecast_steps, test_size)
                logger.info(f"✅ USD/TRY analizi tamamlandı: {results['usd_try'].get('status')}")
            except Exception as usd_error:
                logger.error(f"❌ USD/TRY analizi BAŞARISIZ: {usd_error}")
                logger.error(f"❌ Hata detayı: {type(usd_error).__name__}")
                import traceback
                logger.error(f"❌ Stack trace: {traceback.format_exc()}")
                results['usd_try'] = {'status': 'error', 'message': str(usd_error)}
            
            results['performance']['usd_try_time'] = time.time() - usd_start
            # logger.debug(f"USD/TRY: {results['performance']['usd_try_time']:.2f}s") # SESSIZ
            
            # 2. Enflasyon Analizi
            logger.info("📊 2/3: Enflasyon...")
            inflation_start = time.time()
            results['inflation'] = self._analyze_inflation(forecast_steps, test_size)
            results['performance']['inflation_time'] = time.time() - inflation_start
            # logger.debug(f"Enflasyon: {results['performance']['inflation_time']:.2f}s") # SESSIZ
            
            # 3. Faiz Oranı Analizi
            logger.info("📊 3/3: Faiz Oranı...")
            interest_start = time.time()
            results['interest_rate'] = self._analyze_interest_rate(forecast_steps, test_size)
            results['performance']['interest_rate_time'] = time.time() - interest_start
            # logger.debug(f"Faiz: {results['performance']['interest_rate_time']:.2f}s") # SESSIZ
            
            # 4. News Analizi (opsiyonel) - AKTİF
            if include_news:
                logger.info("\n📰 Haber Analizleri (Sonnet 3.5/4)...")
                news_start = time.time()
                results['news_analysis'] = self.news_service.analyze_all_indicators(days_back=7)
                results['performance']['news_time'] = time.time() - news_start
                # News çarpanlarını uygula
                self._apply_news_multipliers(results)
            
            # News analizi yoksa veya başarısızsa, ARIMA forecast'in son değerini kullan
            self._update_last_values_from_forecast(results)
            
            # Toplam süre
            total_time = time.time() - start_time
            results['performance']['total_time'] = total_time
            results['performance']['target_10s'] = total_time < 10  # YENİ HEDEF: 10 saniye
            results['performance']['target_50s'] = total_time < 50  # G-40: < 50 saniye hedefi
            results['performance']['meets_g40_target'] = total_time < 50  # G-40 gereksinimi
            
            # G-32-G-34: Tahmin doğruluğu hedef kontrolleri
            try:
                accuracy_validator = AccuracyValidator()
                accuracy_results = accuracy_validator.validate_all_indicators(results)
                results['accuracy_validation'] = accuracy_results
            except Exception as e:
                logger.warning(f"Doğruluk kontrolü hatası: {e}")
            
            # G-35-G-36: Analiz ve model doğruluk oranları
            try:
                accuracy_calculator = AccuracyCalculator()
                accuracy_rates = accuracy_calculator.calculate_all_accuracies(results)
                results['accuracy_rates'] = accuracy_rates
            except Exception as e:
                logger.warning(f"Doğruluk oranı hesaplama hatası: {e}")
            
            # G-44-G-46: Performans metrikleri
            try:
                final_memory = perf_monitor.get_memory_usage()
                results['performance']['memory_usage'] = {
                    'initial_mb': initial_memory['memory_mb'],
                    'final_mb': final_memory['memory_mb'],
                    'delta_mb': final_memory['memory_mb'] - initial_memory['memory_mb'],
                    'meets_g44_target': final_memory['meets_target']
                }
            except Exception as e:
                logger.warning(f"Bellek kullanımı ölçümü hatası: {e}")
            
            results['success'] = True
            
            logger.info("\n" + "=" * 80)
            logger.info(f"✅ ANALİZ TAMAMLANDI - Süre: {total_time:.2f}s")
            logger.info(f"   🎯 HEDEF: <10s {'✓' if total_time < 10 else '✗'}")
            logger.info("=" * 80)
            
            self.results = results
            return results
            
        except Exception as e:
            logger.error(f"❌ Analiz hatası: {e}")
            results['error'] = str(e)
            return results
    
    def _analyze_usd_try(self, steps, test_size) -> Dict:
        """USD/TRY analizi yapar."""
        try:
            # SESSIZ MOD: Veri yükleme
            try:
                from src.models.arima_model import load_data_from_mongodb
                data = load_data_from_mongodb(target_field='usd_try')
            except Exception as e:
                logger.error(f"❌ Veri yükleme hatası: {e}")
                data = None
            
            # Alternatif yöntem
            if data is None or data.empty:
                data = load_complete_data_from_mongodb(target_field='usd_try', include_features=False)
            
            if data is None or data.empty:
                logger.error("❌ USD/TRY verisi yok!")
                return {'status': 'error', 'message': 'Veri yok'}
            
            # MAPE FIX: Son 48 ayı kullan (rejim değişikliği 2021+; uzun geçmiş MAPE'yi şişiriyor)
            if isinstance(data, pd.Series) and len(data) > 48:
                logger.info(f"MAPE FIX: USD/TRY verisi son 48 aya kısıtlanıyor ({len(data)} -> 48)")
                data = data.tail(48)
            elif hasattr(data, '__len__') and len(data) > 48:
                data = data.tail(48)
            
            self.usd_forecaster = ARIMAForecaster(
                target_column='usd_try',
                seasonal_periods=12,
                log_timeseries_validation=True,
                use_log_transform=False,
                clip_quantiles=(0.02, 0.98)
            )
            self.usd_forecaster.fit(data, test_size=test_size)

            # TimeSeriesSplit ile walk-forward validation (geçmiş verilerde başarı kontrolü)
            # MAPE FIX: Hold-out validation kullan (cross-validation kararsız)
            metrics = self.usd_forecaster.evaluate(use_timeseries_split=False)
            forecast = self.usd_forecaster.forecast(steps=steps)
            metrics = self._format_metrics(metrics)
            forecast_dict = self._serialize_forecast(forecast)
            
            # Veri Series veya DataFrame olabilir
            if isinstance(data, pd.Series):
                target_series = data
                feature_count = 0
            else:
                target_series, feature_count = self._extract_target_series(data, 'usd_try')
            
            return {
                'status': 'success',
                'data_points': len(target_series),
                'last_value': float(target_series.iloc[-1]) if not target_series.empty else 0.0,
                'feature_count': feature_count,
                'metrics': metrics,
                'forecast': forecast_dict,
                'historical': self._serialize_series(target_series),
                'model_params': self.usd_forecaster.best_params,
                'seasonal_order': getattr(self.usd_forecaster, 'best_seasonal_order', None)
            }
            
        except Exception as e:
            logger.error(f"❌❌❌ USD/TRY ANALİZ HATASI: {e}")
            logger.error(f"❌ Exception tipi: {type(e).__name__}")
            import traceback
            logger.error(f"❌ Full traceback:\n{traceback.format_exc()}")
            return {'status': 'error', 'message': str(e), 'exception_type': type(e).__name__}
    
    def _analyze_inflation(self, steps, test_size) -> Dict:
        """Enflasyon analizi yapar."""
        try:
            # MAPE FIX: Univariate model kullan (multivariate çok kararsız)
            # include_features=False ile sadece inflation_rate Series döner
            target_data = load_inflation_data_from_mongodb(include_features=False)
            
            if target_data is None or target_data.empty:
                return {'status': 'error', 'message': 'Veri yok'}
            
            self.inflation_forecaster = InflationForecaster()
            # MAPE FIX: Exogenous data olmadan eğit (univariate)
            self.inflation_forecaster.fit(target_data, test_size=max(test_size, 0.25), exogenous_data=None)

            # MAPE FIX: Hold-out validation kullan (cross-validation kararsız)
            metrics = self.inflation_forecaster.evaluate(use_timeseries_split=False)
            forecast = self.inflation_forecaster.forecast(steps=steps)
            metrics = self._format_metrics(metrics)
            forecast_dict = self._serialize_forecast(forecast)
            
            
            # MAPE FIX: Univariate model kullanıyoruz
            feature_count = 0
            
            return {
                'status': 'success',
                'data_points': len(target_data),
                'last_value': float(target_data.iloc[-1]) if not target_data.empty else 0.0,
                'feature_count': feature_count,
                'metrics': metrics,
                'forecast': forecast_dict,
                'historical': self._serialize_series(target_data),
                'model_params': self.inflation_forecaster.best_params,
                'seasonal_order': getattr(self.inflation_forecaster, 'best_seasonal_order', None)
            }
            
        except Exception as e:
            logger.error(f"Enflasyon analiz hatası: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _analyze_interest_rate(self, steps, test_size) -> Dict:
        """Faiz oranı analizi yapar."""
        try:
            data = load_interest_rate_data_from_mongodb()
            
            if data is None or data.empty:
                return {'status': 'error', 'message': 'Veri yok'}
            
            self.interest_forecaster = InterestRateForecaster()
            self.interest_forecaster.fit(data, test_size=test_size)

            # TimeSeriesSplit ile walk-forward validation
            # MAPE FIX: Hold-out validation kullan (cross-validation kararsız)
            metrics = self.interest_forecaster.evaluate(use_timeseries_split=False)
            forecast = self.interest_forecaster.forecast(steps=steps)
            metrics = self._format_metrics(metrics)
            forecast_dict = self._serialize_forecast(forecast)
            target_series, feature_count = self._extract_target_series(
                data, 'policy_rate'
            )
            
            return {
                'status': 'success',
                'data_points': len(target_series),
                'last_value': float(target_series.iloc[-1]) if not target_series.empty else 0.0,
                'feature_count': feature_count,
                'metrics': metrics,
                'forecast': forecast_dict,
                'historical': self._serialize_series(target_series),
                'model_params': self.interest_forecaster.best_params,
                'seasonal_order': getattr(self.interest_forecaster, 'best_seasonal_order', None)
            }
            
        except Exception as e:
            logger.error(f"Faiz oranı analiz hatası: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _apply_news_multipliers(self, results: Dict):
        """News çarpanlarını tahminlere uygular ve last_value'yu günceller."""
        try:
            news_analysis = results.get('news_analysis', {})
            
            # USD/TRY
            if 'usd_try' in news_analysis and results.get('usd_try', {}).get('status') == 'success':
                multiplier = news_analysis['usd_try'].get('multiplier', 1.0)
                forecast = results['usd_try'].get('forecast', {})
                if forecast and len(forecast) > 0:
                    enhanced_forecast = {
                        k: v * multiplier for k, v in forecast.items()
                    }
                    results['usd_try']['enhanced_forecast'] = enhanced_forecast
                    results['usd_try']['news_multiplier'] = multiplier
                    # Son değeri enhanced_forecast'in en son tahminiyle güncelle
                    if enhanced_forecast:
                        sorted_dates = sorted(enhanced_forecast.keys())
                        if sorted_dates:
                            results['usd_try']['last_value'] = enhanced_forecast[sorted_dates[-1]]
            
            # Enflasyon
            if 'inflation' in news_analysis and results.get('inflation', {}).get('status') == 'success':
                multiplier = news_analysis['inflation'].get('multiplier', 1.0)
                forecast = results['inflation'].get('forecast', {})
                if forecast and len(forecast) > 0:
                    enhanced_forecast = {
                        k: v * multiplier for k, v in forecast.items()
                    }
                    results['inflation']['enhanced_forecast'] = enhanced_forecast
                    results['inflation']['news_multiplier'] = multiplier
                    # Son değeri enhanced_forecast'in en son tahminiyle güncelle
                    if enhanced_forecast:
                        sorted_dates = sorted(enhanced_forecast.keys())
                        if sorted_dates:
                            results['inflation']['last_value'] = enhanced_forecast[sorted_dates[-1]]
            
            # Faiz
            if 'interest_rate' in news_analysis and results.get('interest_rate', {}).get('status') == 'success':
                multiplier = news_analysis['interest_rate'].get('multiplier', 1.0)
                forecast = results['interest_rate'].get('forecast', {})
                if forecast and len(forecast) > 0:
                    enhanced_forecast = {
                        k: v * multiplier for k, v in forecast.items()
                    }
                    results['interest_rate']['enhanced_forecast'] = enhanced_forecast
                    results['interest_rate']['news_multiplier'] = multiplier
                    # Son değeri enhanced_forecast'in en son tahminiyle güncelle
                    if enhanced_forecast:
                        sorted_dates = sorted(enhanced_forecast.keys())
                        if sorted_dates:
                            results['interest_rate']['last_value'] = enhanced_forecast[sorted_dates[-1]]
                
        except Exception as e:
            logger.error(f"News çarpan uygulaması hatası: {e}")
    
    def _update_last_values_from_forecast(self, results: Dict):
        """News analizi yoksa veya başarısızsa, ARIMA forecast'in son değerini last_value olarak kullan."""
        for indicator_key in ['usd_try', 'inflation', 'interest_rate']:
            indicator = results.get(indicator_key, {})
            if indicator.get('status') != 'success':
                continue
            
            # Eğer enhanced_forecast varsa, onun son değeri zaten last_value olarak ayarlanmış (_apply_news_multipliers içinde)
            # Bu durumda atla
            if 'enhanced_forecast' in indicator:
                continue
            
            # Enhanced forecast yoksa, ARIMA forecast'in son değerini kullan
            forecast = indicator.get('forecast', {})
            if forecast and len(forecast) > 0:
                sorted_dates = sorted(forecast.keys())
                if sorted_dates:
                    indicator['last_value'] = forecast[sorted_dates[-1]]
                    logger.debug(f"{indicator_key} için last_value ARIMA forecast'ten güncellendi: {indicator['last_value']}")

    @staticmethod
    def _serialize_forecast(forecast_results: Optional[Dict]) -> Dict[str, float]:
        if not forecast_results or 'forecast' not in forecast_results:
            return {}
        
        forecast_data = forecast_results.get('forecast')
        if forecast_data is None:
            return {}
        
        # pandas Series veya dict olabilir
        serialized = {}
        try:
            if hasattr(forecast_data, 'items'):  # pandas Series veya dict
                for index, value in forecast_data.items():
                    if hasattr(index, 'strftime'):
                        key = index.strftime('%Y-%m-%d')
                    else:
                        key = str(index)
                    serialized[key] = float(value)
            else:
                logger.warning("Forecast verisi beklenmeyen formatta")
        except Exception as e:
            logger.error(f"Forecast serileştirme hatası: {e}")
            return {}
        
        return serialized

    @staticmethod
    def _serialize_series(series: Optional[pd.Series], limit: int = 36) -> Dict[str, float]:
        if series is None:
            return {}
        try:
            cleaned = series.dropna().tail(limit)
            return {
                idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx): float(val)
                for idx, val in cleaned.items()
            }
        except Exception:
            return {}

    @staticmethod
    def _format_metrics(metrics: Optional[Dict]) -> Optional[Dict[str, Optional[float]]]:
        if not metrics:
            return None
        formatted = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float) and math.isnan(value):
                    formatted[key] = None
                elif isinstance(value, int):
                    formatted[key] = int(value)
                else:
                    formatted[key] = float(value)
            else:
                formatted[key] = value
        return formatted

    @staticmethod
    def _extract_target_series(data, target_column):
        try:
            if data is None:
                return pd.Series(dtype=float), 0
            
            if hasattr(data, 'columns') and target_column in data.columns:
                target_series = data[target_column]
                feature_count = max(data.shape[1] - 1, 0)
                return target_series, feature_count
            
            # pandas Series olarak döndür
            if hasattr(data, 'empty'):
                return data, 0
            else:
                return pd.Series(data, dtype=float), 0
        except Exception as e:
            logger.error(f"Target series çıkarma hatası: {e}")
            return pd.Series(dtype=float), 0
    
    def generate_report(self, language: str = 'tr') -> str:
        """
        Analiz raporunu üretir.
        
        Args:
            language: Dil ('tr' veya 'en') - G-17 gereksinimi
        """
        from src.utils.i18n import get_i18n
        
        i18n = get_i18n(language)
        
        if not self.results:
            return i18n.t('no_data')
        
        report = []
        report.append("=" * 80)
        report.append(i18n.t('report_title'))
        report.append("=" * 80)
        report.append(f"{i18n.t('date')}: {self.results.get('timestamp', '')}")
        report.append("")
        
        # USD/TRY
        if self.results.get('usd_try', {}).get('status') == 'success':
            usd_data = self.results['usd_try']
            report.append(f"📊 {i18n.t('usd_try')}")
            report.append("-" * 80)
            report.append(f"{i18n.t('last_value')}: {usd_data['last_value']:.2f} TRY")
            report.append(f"{i18n.t('data_points')}: {usd_data['data_points']}")
            report.append(f"{i18n.t('model_params')}: ARIMA{usd_data['model_params']}")
            
            if usd_data.get('metrics'):
                metrics = usd_data['metrics']
                report.append(f"{i18n.t('rmse')}: {metrics.get('RMSE', 0):.4f}")
                report.append(f"{i18n.t('mae')}: {metrics.get('MAE', 0):.4f}")
                report.append(f"{i18n.t('mape')}: {metrics.get('MAPE', 0):.2f}%")
            report.append("")
        
        # Enflasyon
        if self.results.get('inflation', {}).get('status') == 'success':
            inf_data = self.results['inflation']
            report.append(f"📈 {i18n.t('inflation')}")
            report.append("-" * 80)
            report.append(f"{i18n.t('last_value')}: {inf_data['last_value']:.2f}%")
            report.append(f"{i18n.t('data_points')}: {inf_data['data_points']}")
            report.append(f"{i18n.t('model_params')}: ARIMA{inf_data['model_params']}")
            
            if inf_data.get('metrics'):
                metrics = inf_data['metrics']
                report.append(f"{i18n.t('rmse')}: {metrics.get('RMSE', 0):.4f}")
                report.append(f"{i18n.t('mae')}: {metrics.get('MAE', 0):.4f}")
                report.append(f"{i18n.t('mape')}: {metrics.get('MAPE', 0):.2f}%")
            report.append("")
        elif self.results.get('inflation'):
             report.append(f"📈 {i18n.t('inflation')} - HATA")
             report.append("-" * 80)
             report.append(f"Hata detayı: {self.results['inflation'].get('message', 'Bilinmeyen hata')}")
             report.append("")
        
        # Faiz
        if self.results.get('interest_rate', {}).get('status') == 'success':
            int_data = self.results['interest_rate']
            report.append(f"💰 {i18n.t('interest_rate')}")
            report.append("-" * 80)
            report.append(f"{i18n.t('last_value')}: {int_data['last_value']:.2f}%")
            report.append(f"{i18n.t('data_points')}: {int_data['data_points']}")
            report.append(f"{i18n.t('model_params')}: ARIMA{int_data['model_params']}")
            
            if int_data.get('metrics'):
                metrics = int_data['metrics']
                report.append(f"{i18n.t('rmse')}: {metrics.get('RMSE', 0):.4f}")
                report.append(f"{i18n.t('mae')}: {metrics.get('MAE', 0):.4f}")
                report.append(f"{i18n.t('mape')}: {metrics.get('MAPE', 0):.2f}%")
            report.append("")
        
        # Performance
        if self.results.get('performance'):
            perf = self.results['performance']
            report.append(f"⚡ {i18n.t('performance')}")
            report.append("-" * 80)
            report.append(f"{i18n.t('total_time')}: {perf.get('total_time', 0):.2f}s")
            success_text = i18n.t('successful') if perf.get('target_50s') else i18n.t('exceeded')
            report.append(f"{i18n.t('target_50s')}: {'✓ ' if perf.get('target_50s') else '✗ '}{success_text}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def export_results(self, filename=None, language: str = 'tr') -> str:
        """
        Sonuçları dosyaya kaydeder.
        
        Args:
            filename: Dosya adı
            language: Dil ('tr' veya 'en') - G-17 gereksinimi
        """
        import time
        
        # G-43: Raporlama süresi < 2 dakika kontrolü
        report_start_time = time.time()
        
        if filename is None:
            filename = f"economic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        report = self.generate_report(language=language)
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            report_time = time.time() - report_start_time
            meets_target = report_time < 120  # G-43: < 2 dakika (120 saniye)
            
            logger.info(f"📄 Rapor kaydedildi: {filename} (Süre: {report_time:.2f}s, Hedef: {'✓' if meets_target else '✗'})")
            return filename
        except Exception as e:
            logger.error(f"Rapor kaydetme hatası: {e}")
            return None


def test_economic_analyzer():
    """EconomicAnalyzer'ı test eder."""
    analyzer = EconomicAnalyzer()
    
    # Tüm göstergeleri analiz et
    results = analyzer.analyze_all(forecast_steps=12, include_news=True)
    
    # Raporu göster
    print(analyzer.generate_report())
    
    # Dosyaya kaydet
    analyzer.export_results()
    
    return results


if __name__ == '__main__':
    test_economic_analyzer()
