"""
Analiz ve Model Doğruluk Oranları Hesaplama
===========================================

TÜBİTAK Projesi - Analiz ve model doğruluk oranları

G-35-G-36 Gereksinimi:
- G-35: Analiz sonuçlarının doğruluk oranı > %95 olmalıdır
- G-36: ARIMA modeli doğruluk oranı %92-95 arası olmalıdır

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

import logging
from typing import Dict, Optional
import numpy as np

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccuracyCalculator:
    """
    Analiz ve model doğruluk oranlarını hesaplayan sınıf.
    """
    
    # G-35-G-36: Hedef doğruluk oranları
    TARGET_ANALYSIS_ACCURACY = 0.95  # G-35: > %95
    TARGET_ARIMA_ACCURACY_MIN = 0.92  # G-36: %92-95 arası
    TARGET_ARIMA_ACCURACY_MAX = 0.95  # G-36: %92-95 arası
    
    def __init__(self):
        """AccuracyCalculator'ı başlatır."""
        logger.info("AccuracyCalculator başlatıldı.")
    
    def calculate_model_accuracy(self, metrics: Dict) -> float:
        """
        Model doğruluğunu hesaplar (MAPE'den).
        
        Args:
            metrics: Performans metrikleri (MAPE içermeli)
            
        Returns:
            float: Doğruluk oranı (0-1 arası)
        """
        mape = metrics.get('MAPE', 100.0)
        accuracy = max(0.0, 1.0 - (mape / 100.0))
        return accuracy
    
    def calculate_analysis_accuracy(self, results: Dict) -> Dict:
        """
        Analiz sonuçlarının genel doğruluğunu hesaplar.
        
        G-35 gereksinimi: Analiz sonuçlarının doğruluk oranı > %95
        
        Args:
            results: Analiz sonuçları
            
        Returns:
            Dict: Analiz doğruluğu metrikleri
        """
        indicators = ['usd_try', 'inflation', 'interest_rate']
        accuracies = []
        
        for indicator in indicators:
            if indicator not in results:
                continue
            
            indicator_data = results[indicator]
            if indicator_data.get('status') != 'success':
                continue
            
            metrics = indicator_data.get('metrics', {})
            if not metrics:
                continue
            
            accuracy = self.calculate_model_accuracy(metrics)
            accuracies.append(accuracy)
        
        if not accuracies:
            return {
                'overall_accuracy': 0.0,
                'meets_target': False,
                'target': self.TARGET_ANALYSIS_ACCURACY
            }
        
        overall_accuracy = np.mean(accuracies)
        meets_target = overall_accuracy >= self.TARGET_ANALYSIS_ACCURACY
        
        result = {
            'overall_accuracy': float(overall_accuracy),
            'overall_accuracy_percentage': float(overall_accuracy * 100),
            'individual_accuracies': {ind: acc for ind, acc in zip(indicators[:len(accuracies)], accuracies)},
            'target': self.TARGET_ANALYSIS_ACCURACY,
            'target_percentage': self.TARGET_ANALYSIS_ACCURACY * 100,
            'meets_target': meets_target,
            'status': 'PASS' if meets_target else 'FAIL'
        }
        
        status_icon = '✅' if meets_target else '❌'
        logger.info(
            f"{status_icon} Analiz doğruluğu: {overall_accuracy*100:.2f}% "
            f"(Hedef: {self.TARGET_ANALYSIS_ACCURACY*100:.0f}%) - {'BAŞARILI' if meets_target else 'HEDEF AŞILDI'}"
        )
        
        return result
    
    def validate_arima_accuracy(self, metrics: Dict) -> Dict:
        """
        ARIMA model doğruluğunu validate eder.
        
        G-36 gereksinimi: ARIMA modeli doğruluk oranı %92-95 arası olmalıdır
        
        Args:
            metrics: ARIMA model metrikleri
            
        Returns:
            Dict: ARIMA doğruluk validasyonu sonuçları
        """
        accuracy = self.calculate_model_accuracy(metrics)
        
        # G-36: %92-95 arası kontrolü
        meets_min = accuracy >= self.TARGET_ARIMA_ACCURACY_MIN
        meets_max = accuracy <= self.TARGET_ARIMA_ACCURACY_MAX
        in_range = meets_min and meets_max
        
        result = {
            'accuracy': accuracy,
            'accuracy_percentage': accuracy * 100,
            'target_min': self.TARGET_ARIMA_ACCURACY_MIN,
            'target_max': self.TARGET_ARIMA_ACCURACY_MAX,
            'meets_min': meets_min,
            'meets_max': meets_max,
            'in_range': in_range,
            'status': 'PASS' if in_range else 'FAIL'
        }
        
        status_icon = '✅' if in_range else '❌'
        logger.info(
            f"{status_icon} ARIMA doğruluğu: {accuracy*100:.2f}% "
            f"(Hedef: {self.TARGET_ARIMA_ACCURACY_MIN*100:.0f}%-{self.TARGET_ARIMA_ACCURACY_MAX*100:.0f}%) - "
            f"{'BAŞARILI' if in_range else 'HEDEF DIŞINDA'}"
        )
        
        return result
    
    def calculate_all_accuracies(self, results: Dict) -> Dict:
        """
        Tüm doğruluk metriklerini hesaplar.
        
        Args:
            results: Analiz sonuçları
            
        Returns:
            Dict: Tüm doğruluk metrikleri
        """
        # G-35: Analiz doğruluğu
        analysis_accuracy = self.calculate_analysis_accuracy(results)
        
        # G-36: ARIMA doğrulukları
        arima_accuracies = {}
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
            
            arima_accuracies[indicator] = self.validate_arima_accuracy(metrics)
        
        return {
            'analysis_accuracy': analysis_accuracy,  # G-35
            'arima_accuracies': arima_accuracies,    # G-36
            'summary': {
                'analysis_meets_target': analysis_accuracy.get('meets_target', False),
                'arima_models_in_range': sum(
                    1 for acc in arima_accuracies.values() 
                    if acc.get('in_range', False)
                ),
                'total_arima_models': len(arima_accuracies)
            }
        }


def calculate_accuracy_rates(results: Dict) -> Dict:
    """
    Doğruluk oranlarını hesaplamak için kolay kullanım fonksiyonu.
    
    Args:
        results: Analiz sonuçları
        
    Returns:
        Dict: Doğruluk metrikleri
    """
    calculator = AccuracyCalculator()
    return calculator.calculate_all_accuracies(results)


if __name__ == '__main__':
    # Test
    logger.info("AccuracyCalculator test ediliyor...")
    
    calculator = AccuracyCalculator()
    
    # Test sonuçları
    test_results = {
        'usd_try': {
            'status': 'success',
            'metrics': {'MAPE': 8.0}  # %92 doğruluk
        },
        'inflation': {
            'status': 'success',
            'metrics': {'MAPE': 5.0}  # %95 doğruluk
        },
        'interest_rate': {
            'status': 'success',
            'metrics': {'MAPE': 6.0}  # %94 doğruluk
        }
    }
    
    accuracies = calculator.calculate_all_accuracies(test_results)
    logger.info(f"Doğruluk metrikleri: {accuracies}")
    
    logger.info("AccuracyCalculator test tamamlandı.")

