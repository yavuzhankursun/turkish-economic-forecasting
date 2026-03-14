"""
Tahmin Doğruluğu Hedef Kontrolü
=================================

TÜBİTAK Projesi - Tahmin doğruluğu hedef kontrolleri

G-32-G-34 Gereksinimi: 
- G-32: Döviz kuru tahmin doğruluğu > %85 olmalıdır
- G-33: Enflasyon tahmin doğruluğu > %80 olmalıdır
- G-34: Faiz oranı tahmin doğruluğu > %90 olmalıdır

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

import logging
from typing import Dict, Optional
import numpy as np

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccuracyValidator:
    """
    Tahmin doğruluğu hedef kontrollerini yapan sınıf.
    """
    
    # G-32-G-34: Hedef doğruluk oranları
    TARGET_ACCURACY = {
        'usd_try': 0.85,      # G-32: > %85
        'inflation': 0.80,    # G-33: > %80
        'interest_rate': 0.90  # G-34: > %90
    }
    
    def __init__(self):
        """AccuracyValidator'ı başlatır."""
        logger.info("AccuracyValidator başlatıldı.")
    
    def calculate_accuracy(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Tahmin doğruluğunu hesaplar (1 - MAPE/100).
        
        Args:
            actual: Gerçek değerler
            predicted: Tahmin edilen değerler
            
        Returns:
            float: Doğruluk oranı (0-1 arası)
        """
        if len(actual) == 0 or len(predicted) == 0:
            return 0.0
        
        # Uzunlukları eşitle
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        # MAPE hesapla
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
        
        # Doğruluk = 1 - (MAPE / 100), ancak MAPE çok yüksekse negatif olabilir
        accuracy = max(0.0, 1.0 - (mape / 100.0))
        
        return accuracy
    
    def validate_accuracy(self, indicator_type: str, metrics: Dict) -> Dict[str, any]:
        """
        Gösterge için doğruluğu kontrol eder.
        
        Args:
            indicator_type: Gösterge tipi ('usd_try', 'inflation', 'interest_rate')
            metrics: Performans metrikleri (MAPE içermeli)
            
        Returns:
            Dict: Doğruluk kontrolü sonuçları
        """
        target = self.TARGET_ACCURACY.get(indicator_type, 0.80)
        
        # MAPE'den doğruluğu hesapla
        mape = metrics.get('MAPE', 100.0)
        accuracy = max(0.0, 1.0 - (mape / 100.0))
        
        # Hedef kontrolü
        meets_target = accuracy >= target
        
        result = {
            'indicator_type': indicator_type,
            'accuracy': accuracy,
            'accuracy_percentage': accuracy * 100,
            'target': target,
            'target_percentage': target * 100,
            'meets_target': meets_target,
            'mape': mape,
            'status': 'PASS' if meets_target else 'FAIL'
        }
        
        status_icon = '✅' if meets_target else '❌'
        logger.info(
            f"{status_icon} {indicator_type}: Doğruluk={accuracy*100:.2f}% "
            f"(Hedef: {target*100:.0f}%) - {'BAŞARILI' if meets_target else 'HEDEF AŞILDI'}"
        )
        
        return result
    
    def validate_all_indicators(self, results: Dict) -> Dict[str, Dict]:
        """
        Tüm göstergeler için doğruluğu kontrol eder.
        
        Args:
            results: Analiz sonuçları
            
        Returns:
            Dict: Tüm göstergeler için doğruluk kontrolleri
        """
        validation_results = {}
        
        indicators = ['usd_try', 'inflation', 'interest_rate']
        
        for indicator in indicators:
            if indicator not in results:
                continue
            
            indicator_data = results[indicator]
            if indicator_data.get('status') != 'success':
                continue
            
            metrics = indicator_data.get('metrics', {})
            if not metrics:
                continue
            
            validation_results[indicator] = self.validate_accuracy(indicator, metrics)
        
        # Genel durum
        all_pass = all(
            v.get('meets_target', False) 
            for v in validation_results.values()
        )
        
        validation_results['overall'] = {
            'all_targets_met': all_pass,
            'total_indicators': len(validation_results),
            'passed': sum(1 for v in validation_results.values() if v.get('meets_target', False))
        }
        
        logger.info(f"Genel doğruluk kontrolü: {validation_results['overall']['passed']}/{validation_results['overall']['total_indicators']} hedef karşılandı")
        
        return validation_results


def validate_forecast_accuracy(indicator_type: str, metrics: Dict) -> Dict:
    """
    Tahmin doğruluğunu validate etmek için kolay kullanım fonksiyonu.
    
    Args:
        indicator_type: Gösterge tipi
        metrics: Performans metrikleri
        
    Returns:
        Dict: Doğruluk kontrolü sonuçları
    """
    validator = AccuracyValidator()
    return validator.validate_accuracy(indicator_type, metrics)


if __name__ == '__main__':
    # Test
    logger.info("AccuracyValidator test ediliyor...")
    
    validator = AccuracyValidator()
    
    # Test metrikleri
    test_metrics_usd = {'MAPE': 10.0}  # %90 doğruluk
    test_metrics_inf = {'MAPE': 25.0}  # %75 doğruluk (hedef: %80)
    test_metrics_int = {'MAPE': 5.0}   # %95 doğruluk
    
    result_usd = validator.validate_accuracy('usd_try', test_metrics_usd)
    result_inf = validator.validate_accuracy('inflation', test_metrics_inf)
    result_int = validator.validate_accuracy('interest_rate', test_metrics_int)
    
    logger.info(f"USD/TRY: {result_usd}")
    logger.info(f"Inflation: {result_inf}")
    logger.info(f"Interest Rate: {result_int}")
    
    logger.info("AccuracyValidator test tamamlandı.")

