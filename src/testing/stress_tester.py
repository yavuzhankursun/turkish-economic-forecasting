"""
Stres Testi Modülü
==================

TÜBİTAK Projesi - Model kararlılığını stres testi ile değerlendirme

G-38 Gereksinimi: Sistemin stres testi kararlılığı > %99 olmalıdır.

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Callable
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StressTester:
    """
    Stres testi sınıfı.
    
    Özellikler:
    - Veri gürültüsü testi
    - Eksik veri testi
    - Aykırı değer testi
    - Kararlılık metrikleri
    """
    
    TARGET_STABILITY = 0.99  # G-38: > %99
    
    def __init__(self):
        """StressTester'ı başlatır."""
        logger.info("StressTester başlatıldı.")
    
    def add_noise(self, data: pd.Series, noise_level: float = 0.1) -> pd.Series:
        """
        Veriye gürültü ekler.
        
        Args:
            data: Orijinal veri
            noise_level: Gürültü seviyesi (0-1 arası)
            
        Returns:
            pd.Series: Gürültülü veri
        """
        noise = np.random.normal(0, noise_level * data.std(), len(data))
        noisy_data = data + noise
        return noisy_data
    
    def remove_random_values(self, data: pd.Series, missing_rate: float = 0.1) -> pd.Series:
        """
        Rastgele değerleri kaldırır (eksik veri simülasyonu).
        
        Args:
            data: Orijinal veri
            missing_rate: Eksik veri oranı (0-1 arası)
            
        Returns:
            pd.Series: Eksik değerler içeren veri
        """
        data_copy = data.copy()
        n_missing = int(len(data) * missing_rate)
        missing_indices = np.random.choice(len(data), n_missing, replace=False)
        data_copy.iloc[missing_indices] = np.nan
        return data_copy
    
    def add_outliers(self, data: pd.Series, outlier_rate: float = 0.05, 
                    outlier_multiplier: float = 3.0) -> pd.Series:
        """
        Aykırı değerler ekler.
        
        Args:
            data: Orijinal veri
            outlier_rate: Aykırı değer oranı (0-1 arası)
            outlier_multiplier: Aykırı değer çarpanı
            
        Returns:
            pd.Series: Aykırı değerler içeren veri
        """
        data_copy = data.copy()
        n_outliers = int(len(data) * outlier_rate)
        outlier_indices = np.random.choice(len(data), n_outliers, replace=False)
        
        for idx in outlier_indices:
            if np.random.rand() > 0.5:
                data_copy.iloc[idx] = data.iloc[idx] * outlier_multiplier
            else:
                data_copy.iloc[idx] = data.iloc[idx] / outlier_multiplier
        
        return data_copy
    
    def stress_test(self, data: pd.Series,
                   model_trainer: Callable,
                   model_predictor: Callable,
                   scenarios: Optional[List[Dict]] = None) -> Dict:
        """
        Stres testi yapar.
        
        G-38 gereksinimi: Stres testi kararlılığı > %99
        
        Args:
            data: Orijinal veri
            model_trainer: Model eğitme fonksiyonu
            model_predictor: Model tahmin fonksiyonu
            scenarios: Özel senaryolar (None ise varsayılan senaryolar kullanılır)
            
        Returns:
            Dict: Stres testi sonuçları
        """
        if scenarios is None:
            scenarios = [
                {'name': 'baseline', 'noise': 0.0, 'missing': 0.0, 'outliers': 0.0},
                {'name': 'low_noise', 'noise': 0.05, 'missing': 0.0, 'outliers': 0.0},
                {'name': 'medium_noise', 'noise': 0.1, 'missing': 0.0, 'outliers': 0.0},
                {'name': 'high_noise', 'noise': 0.2, 'missing': 0.0, 'outliers': 0.0},
                {'name': 'missing_5pct', 'noise': 0.0, 'missing': 0.05, 'outliers': 0.0},
                {'name': 'missing_10pct', 'noise': 0.0, 'missing': 0.1, 'outliers': 0.0},
                {'name': 'outliers_5pct', 'noise': 0.0, 'missing': 0.0, 'outliers': 0.05},
                {'name': 'combined', 'noise': 0.1, 'missing': 0.05, 'outliers': 0.05}
            ]
        
        baseline_predictions = None
        scenario_results = []
        
        logger.info(f"Stres testi başlatılıyor: {len(scenarios)} senaryo")
        
        for scenario in scenarios:
            scenario_name = scenario['name']
            logger.info(f"Senaryo: {scenario_name}")
            
            # Veriyi boz
            test_data = data.copy()
            
            if scenario.get('noise', 0) > 0:
                test_data = self.add_noise(test_data, scenario['noise'])
            
            if scenario.get('missing', 0) > 0:
                test_data = self.remove_random_values(test_data, scenario['missing'])
                test_data = test_data.fillna(method='ffill').fillna(method='bfill')
            
            if scenario.get('outliers', 0) > 0:
                test_data = self.add_outliers(test_data, scenario['outliers'])
            
            try:
                # Modeli eğit
                model = model_trainer(test_data)
                
                # Tahmin yap
                predictions = model_predictor(model, test_data, steps=12)
                
                if isinstance(predictions, pd.Series):
                    pred_values = predictions.values
                else:
                    pred_values = predictions
                
                # Baseline ile karşılaştır
                if baseline_predictions is None:
                    baseline_predictions = pred_values
                    scenario_results.append({
                        'scenario': scenario_name,
                        'status': 'baseline',
                        'predictions': pred_values.tolist()
                    })
                else:
                    # Baseline ile farkı hesapla
                    if len(pred_values) == len(baseline_predictions):
                        diff = np.abs(pred_values - baseline_predictions)
                        relative_diff = diff / (np.abs(baseline_predictions) + 1e-10)
                        stability = 1.0 - np.mean(relative_diff)
                        stability = max(0.0, min(1.0, stability))  # 0-1 arasına sınırla
                    else:
                        stability = 0.0
                    
                    scenario_results.append({
                        'scenario': scenario_name,
                        'status': 'tested',
                        'stability': float(stability),
                        'stability_percentage': float(stability * 100),
                        'predictions': pred_values.tolist()
                    })
                    
                    logger.info(f"  {scenario_name}: Kararlılık={stability*100:.2f}%")
                
            except Exception as e:
                logger.warning(f"Senaryo {scenario_name} başarısız: {e}")
                scenario_results.append({
                    'scenario': scenario_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Genel kararlılık hesapla
        tested_scenarios = [r for r in scenario_results if r.get('status') == 'tested']
        if tested_scenarios:
            stabilities = [r['stability'] for r in tested_scenarios]
            overall_stability = np.mean(stabilities)
            meets_target = overall_stability >= self.TARGET_STABILITY
        else:
            overall_stability = 0.0
            meets_target = False
        
        results = {
            'overall_stability': float(overall_stability),
            'overall_stability_percentage': float(overall_stability * 100),
            'target': self.TARGET_STABILITY,
            'target_percentage': self.TARGET_STABILITY * 100,
            'meets_target': meets_target,
            'scenario_results': scenario_results,
            'n_scenarios': len(scenarios),
            'n_tested': len(tested_scenarios),
            'status': 'PASS' if meets_target else 'FAIL'
        }
        
        status_icon = '✅' if meets_target else '❌'
        logger.info(
            f"{status_icon} Stres testi kararlılığı: {overall_stability*100:.2f}% "
            f"(Hedef: {self.TARGET_STABILITY*100:.0f}%) - {'BAŞARILI' if meets_target else 'HEDEF AŞILDI'}"
        )
        
        return results


def stress_test_model(data: pd.Series, model_trainer: Callable, model_predictor: Callable,
                     scenarios: Optional[List[Dict]] = None) -> Dict:
    """
    Model stres testi yapmak için kolay kullanım fonksiyonu.
    
    Args:
        data: Zaman serisi verisi
        model_trainer: Model eğitme fonksiyonu
        model_predictor: Model tahmin fonksiyonu
        scenarios: Özel senaryolar
        
    Returns:
        Dict: Stres testi sonuçları
    """
    tester = StressTester()
    return tester.stress_test(data, model_trainer, model_predictor, scenarios)


if __name__ == '__main__':
    # Test
    logger.info("StressTester test ediliyor...")
    
    # Örnek veri
    dates = pd.date_range('2024-01-01', periods=100, freq='MS')
    data = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)
    
    # Basit model
    def simple_trainer(train_data):
        class SimpleModel:
            def predict(self, steps):
                return np.random.randn(steps).cumsum() + train_data.iloc[-1]
        return SimpleModel()
    
    def simple_predictor(model, train_data, steps):
        return model.predict(steps)
    
    tester = StressTester()
    results = tester.stress_test(data, simple_trainer, simple_predictor)
    
    logger.info(f"Stres testi sonuçları: Kararlılık={results['overall_stability']*100:.2f}%")
    logger.info("StressTester test tamamlandı.")

