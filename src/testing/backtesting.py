"""
Geriye Dönük Test (Back-testing) Modülü
========================================

TÜBİTAK Projesi - Model performansını geriye dönük test ile değerlendirme

G-37 Gereksinimi: Sistemin geriye dönük test (back-testing) doğruluğu > %85 olmalıdır.

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """
    Geriye dönük test (back-testing) sınıfı.
    
    Özellikler:
    - Walk-forward back-testing
    - Rolling window back-testing
    - Doğruluk metrikleri hesaplama
    """
    
    TARGET_ACCURACY = 0.85  # G-37: > %85
    
    def __init__(self):
        """Backtester'ı başlatır."""
        logger.info("Backtester başlatıldı.")
    
    def walk_forward_backtest(self, data: pd.Series, 
                             model_trainer: Callable,
                             model_predictor: Callable,
                             train_window: int = 24,
                             test_window: int = 6,
                             step_size: int = 1) -> Dict:
        """
        Walk-forward back-testing yapar.
        
        Args:
            data: Zaman serisi verisi
            model_trainer: Model eğitme fonksiyonu
            model_predictor: Model tahmin fonksiyonu
            train_window: Eğitim penceresi boyutu
            test_window: Test penceresi boyutu
            step_size: Her iterasyonda kaydırma miktarı
            
        Returns:
            Dict: Back-testing sonuçları
        """
        if len(data) < train_window + test_window:
            raise ValueError(f"Yetersiz veri: {len(data)} < {train_window + test_window}")
        
        all_predictions = []
        all_actuals = []
        all_dates = []
        
        logger.info(f"Walk-forward back-testing başlatılıyor: train={train_window}, test={test_window}")
        
        # Walk-forward loop
        start_idx = 0
        iteration = 0
        
        while start_idx + train_window + test_window <= len(data):
            # Train ve test setlerini ayır
            train_end = start_idx + train_window
            test_start = train_end
            test_end = min(test_start + test_window, len(data))
            
            train_data = data.iloc[start_idx:train_end]
            test_data = data.iloc[test_start:test_end]
            
            try:
                # Modeli eğit
                model = model_trainer(train_data)
                
                # Tahmin yap
                predictions = model_predictor(model, train_data, len(test_data))
                
                # Sonuçları topla
                if isinstance(predictions, pd.Series):
                    pred_values = predictions.values
                    pred_dates = predictions.index
                else:
                    pred_values = predictions
                    pred_dates = test_data.index[:len(pred_values)]
                
                actual_values = test_data.iloc[:len(pred_values)].values
                actual_dates = test_data.index[:len(pred_values)]
                
                all_predictions.extend(pred_values)
                all_actuals.extend(actual_values)
                all_dates.extend(actual_dates)
                
                iteration += 1
                logger.info(f"Iterasyon {iteration}: {len(pred_values)} tahmin")
                
            except Exception as e:
                logger.warning(f"Iterasyon {iteration} başarısız: {e}")
            
            # Bir sonraki iterasyon için kaydır
            start_idx += step_size
        
        if not all_predictions:
            raise ValueError("Hiçbir iterasyon başarılı olmadı")
        
        # Metrikleri hesapla
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        
        mae = mean_absolute_error(all_actuals, all_predictions)
        mse = mean_squared_error(all_actuals, all_predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((all_actuals - all_predictions) / (all_actuals + 1e-10))) * 100
        
        # Doğruluk (G-37: > %85)
        accuracy = max(0.0, 1.0 - (mape / 100.0))
        meets_target = accuracy >= self.TARGET_ACCURACY
        
        results = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'accuracy': float(accuracy),
            'accuracy_percentage': float(accuracy * 100),
            'target': self.TARGET_ACCURACY,
            'meets_target': meets_target,
            'n_iterations': iteration,
            'n_predictions': len(all_predictions),
            'status': 'PASS' if meets_target else 'FAIL'
        }
        
        status_icon = '✅' if meets_target else '❌'
        logger.info(
            f"{status_icon} Back-testing doğruluğu: {accuracy*100:.2f}% "
            f"(Hedef: {self.TARGET_ACCURACY*100:.0f}%) - {'BAŞARILI' if meets_target else 'HEDEF AŞILDI'}"
        )
        
        return results
    
    def rolling_window_backtest(self, data: pd.Series,
                               model_trainer: Callable,
                               model_predictor: Callable,
                               window_size: int = 24,
                               forecast_horizon: int = 6) -> Dict:
        """
        Rolling window back-testing yapar.
        
        Args:
            data: Zaman serisi verisi
            model_trainer: Model eğitme fonksiyonu
            model_predictor: Model tahmin fonksiyonu
            window_size: Pencere boyutu
            forecast_horizon: Tahmin ufku
            
        Returns:
            Dict: Back-testing sonuçları
        """
        if len(data) < window_size + forecast_horizon:
            raise ValueError(f"Yetersiz veri: {len(data)} < {window_size + forecast_horizon}")
        
        all_predictions = []
        all_actuals = []
        
        logger.info(f"Rolling window back-testing başlatılıyor: window={window_size}, horizon={forecast_horizon}")
        
        # Rolling window loop
        for i in range(len(data) - window_size - forecast_horizon + 1):
            train_data = data.iloc[i:i+window_size]
            test_data = data.iloc[i+window_size:i+window_size+forecast_horizon]
            
            try:
                # Modeli eğit
                model = model_trainer(train_data)
                
                # Tahmin yap
                predictions = model_predictor(model, train_data, len(test_data))
                
                # Sonuçları topla
                if isinstance(predictions, pd.Series):
                    pred_values = predictions.values
                else:
                    pred_values = predictions
                
                actual_values = test_data.values[:len(pred_values)]
                
                all_predictions.extend(pred_values)
                all_actuals.extend(actual_values)
                
            except Exception as e:
                logger.warning(f"Pencere {i} başarısız: {e}")
                continue
        
        if not all_predictions:
            raise ValueError("Hiçbir pencere başarılı olmadı")
        
        # Metrikleri hesapla
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        
        mae = mean_absolute_error(all_actuals, all_predictions)
        mse = mean_squared_error(all_actuals, all_predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((all_actuals - all_predictions) / (all_actuals + 1e-10))) * 100
        
        # Doğruluk
        accuracy = max(0.0, 1.0 - (mape / 100.0))
        meets_target = accuracy >= self.TARGET_ACCURACY
        
        results = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'accuracy': float(accuracy),
            'accuracy_percentage': float(accuracy * 100),
            'target': self.TARGET_ACCURACY,
            'meets_target': meets_target,
            'n_windows': len(data) - window_size - forecast_horizon + 1,
            'n_predictions': len(all_predictions),
            'status': 'PASS' if meets_target else 'FAIL'
        }
        
        logger.info(f"Rolling window back-testing tamamlandı: Doğruluk={accuracy*100:.2f}%")
        return results


def backtest_model(data: pd.Series, model_trainer: Callable, model_predictor: Callable,
                  method: str = 'walk_forward', **kwargs) -> Dict:
    """
    Model back-testing yapmak için kolay kullanım fonksiyonu.
    
    Args:
        data: Zaman serisi verisi
        model_trainer: Model eğitme fonksiyonu
        model_predictor: Model tahmin fonksiyonu
        method: Back-testing yöntemi ('walk_forward', 'rolling_window')
        **kwargs: Back-testing parametreleri
        
    Returns:
        Dict: Back-testing sonuçları
    """
    backtester = Backtester()
    
    if method == 'walk_forward':
        return backtester.walk_forward_backtest(data, model_trainer, model_predictor, **kwargs)
    elif method == 'rolling_window':
        return backtester.rolling_window_backtest(data, model_trainer, model_predictor, **kwargs)
    else:
        raise ValueError(f"Bilinmeyen yöntem: {method}")


if __name__ == '__main__':
    # Test
    logger.info("Backtester test ediliyor...")
    
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
    
    backtester = Backtester()
    results = backtester.walk_forward_backtest(
        data, simple_trainer, simple_predictor,
        train_window=24, test_window=6
    )
    
    logger.info(f"Back-testing sonuçları: {results}")
    logger.info("Backtester test tamamlandı.")

