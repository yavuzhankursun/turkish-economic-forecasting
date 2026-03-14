"""
Model Validasyon Modülü
=======================

TÜBİTAK Projesi - Cross-validation ve hold-out validasyonu

G-30 Gereksinimi: Modelin geçerliliği için çapraz doğrulama (cross-validation) 
ve hold-out validasyonu teknikleri kullanılmalıdır.

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Model validasyon sınıfı.
    
    Özellikler:
    - Hold-out validasyonu
    - Time series cross-validation (walk-forward)
    - K-fold cross-validation (zaman serisi için uyarlanmış)
    """
    
    def __init__(self):
        """ModelValidator'ı başlatır."""
        logger.info("ModelValidator başlatıldı.")
    
    def hold_out_validation(self, data: pd.Series, model_trainer, model_predictor,
                          test_size: float = 0.2) -> Dict[str, float]:
        """
        Hold-out validasyonu yapar.
        
        G-30 gereksinimi: Hold-out validasyonu
        
        Args:
            data: Zaman serisi verisi
            test_size: Test seti oranı
            model_trainer: Model eğitme fonksiyonu
            model_predictor: Model tahmin fonksiyonu
            
        Returns:
            Dict[str, float]: Validasyon metrikleri
        """
        if len(data) < 10:
            raise ValueError("Yetersiz veri için hold-out validasyonu yapılamaz")
        
        # Train-test split
        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        logger.info(f"Hold-out validasyonu: Train={len(train_data)}, Test={len(test_data)}")
        
        # Modeli eğit
        model = model_trainer(train_data)
        
        # Tahmin yap
        predictions = model_predictor(model, train_data, len(test_data))
        
        # Metrikleri hesapla
        if isinstance(predictions, pd.Series):
            pred_values = predictions.values
        else:
            pred_values = predictions
        
        actual_values = test_data.values
        
        # Uzunlukları eşitle
        min_len = min(len(pred_values), len(actual_values))
        pred_values = pred_values[:min_len]
        actual_values = actual_values[:min_len]
        
        mae = mean_absolute_error(actual_values, pred_values)
        mse = mean_squared_error(actual_values, pred_values)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual_values - pred_values) / (actual_values + 1e-10))) * 100
        
        metrics = {
            'MAE': float(mae),
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAPE': float(mape),
            'train_size': len(train_data),
            'test_size': len(test_data)
        }
        
        logger.info(f"Hold-out validasyon sonuçları: RMSE={rmse:.4f}, MAPE={mape:.2f}%")
        return metrics
    
    def time_series_cross_validation(self, data: pd.Series, model_trainer, model_predictor,
                                    n_splits: int = 5) -> Dict[str, float]:
        """
        Zaman serisi için cross-validation yapar (walk-forward).
        
        G-30 gereksinimi: Cross-validation
        
        Args:
            data: Zaman serisi verisi
            n_splits: Fold sayısı
            model_trainer: Model eğitme fonksiyonu
            model_predictor: Model tahmin fonksiyonu
            
        Returns:
            Dict[str, float]: Cross-validation metrikleri
        """
        # Minimum veri gereksinimi: Her fold için en az 3 test verisi, train için en az 12 veri
        min_train_size = 12
        min_test_size = 3
        min_total_size = min_train_size + (n_splits * min_test_size)
        
        if len(data) < min_total_size:
            # Veri azsa fold sayısını azalt
            max_splits = max(1, (len(data) - min_train_size) // min_test_size)
            if max_splits < n_splits:
                logger.warning(f"Veri yetersiz ({len(data)} nokta). Fold sayısı {n_splits}'den {max_splits}'e düşürülüyor.")
                n_splits = max_splits
        
        if n_splits < 1:
            raise ValueError(f"Yetersiz veri: {len(data)} < {min_total_size}")
        
        # Fold'ları oluştur - daha dengeli dağılım için
        # Her fold için minimum train ve test boyutlarını garanti et
        total_available = len(data) - min_train_size
        fold_size = max(min_test_size, total_available // (n_splits + 1))
        all_metrics = []
        
        logger.info(f"Time series cross-validation başlatılıyor: {n_splits} fold (veri: {len(data)} nokta, fold_size: ~{fold_size})")
        
        for i in range(n_splits):
            # Train: ilk min_train_size + i*fold_size, Test: train_end'den train_end + fold_size'ye
            train_end = min_train_size + (i + 1) * fold_size
            test_start = train_end
            test_end = min(train_end + fold_size, len(data))
            
            # Minimum test boyutu kontrolü
            if test_end - test_start < min_test_size:
                logger.warning(f"Fold {i+1}: Test seti çok küçük ({test_end - test_start} < {min_test_size}), atlanıyor.")
                continue
            
            if train_end < min_train_size:
                logger.warning(f"Fold {i+1}: Train seti çok küçük ({train_end} < {min_train_size}), atlanıyor.")
                continue
            
            train_data = data.iloc[:train_end]
            test_data = data.iloc[test_start:test_end]
            
            try:
                # Modeli eğit
                model = model_trainer(train_data)
                
                # Tahmin yap
                predictions = model_predictor(model, train_data, len(test_data))
                
                # Metrikleri hesapla
                if isinstance(predictions, pd.Series):
                    pred_values = predictions.values
                else:
                    pred_values = predictions
                
                actual_values = test_data.values
                
                # Uzunlukları eşitle
                min_len = min(len(pred_values), len(actual_values))
                if min_len < 1:
                    continue
                
                pred_values = pred_values[:min_len]
                actual_values = actual_values[:min_len]
                
                # NaN ve Inf kontrolü
                mask = ~np.isnan(pred_values) & ~np.isinf(pred_values) & ~np.isnan(actual_values) & ~np.isinf(actual_values)
                pred_values = pred_values[mask]
                actual_values = actual_values[mask]
                
                if len(pred_values) < 1:
                    logger.warning(f"Fold {i+1}: Geçerli (NaN olmayan) tahmin yok, atlanıyor.")
                    continue
                
                mae = mean_absolute_error(actual_values, pred_values)
                mse = mean_squared_error(actual_values, pred_values)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((actual_values - pred_values) / (actual_values + 1e-10))) * 100
                
                all_metrics.append({
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAPE': mape
                })
                
                logger.info(f"Fold {i+1}/{n_splits}: RMSE={rmse:.4f}, MAPE={mape:.2f}%")
                
            except Exception as e:
                logger.warning(f"Fold {i+1} başarısız: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        if not all_metrics:
            raise ValueError("Hiçbir fold başarıyla tamamlanamadı")
        
        # Ortalama metrikleri hesapla
        avg_metrics = {
            'MAE': np.mean([m['MAE'] for m in all_metrics]),
            'MSE': np.mean([m['MSE'] for m in all_metrics]),
            'RMSE': np.mean([m['RMSE'] for m in all_metrics]),
            'MAPE': np.mean([m['MAPE'] for m in all_metrics]),
            'std_RMSE': np.std([m['RMSE'] for m in all_metrics]),
            'std_MAPE': np.std([m['MAPE'] for m in all_metrics]),
            'n_splits': len(all_metrics)
        }
        
        logger.info(f"Cross-validation tamamlandı: Ortalama RMSE={avg_metrics['RMSE']:.4f}, MAPE={avg_metrics['MAPE']:.2f}%")
        return avg_metrics
    
    def validate_model(self, data: pd.Series, model_trainer, model_predictor,
                      validation_type: str = 'both', test_size: float = 0.2,
                      n_splits: int = 5) -> Dict[str, Dict]:
        """
        Model validasyonu yapar.
        
        Args:
            data: Zaman serisi verisi
            model_trainer: Model eğitme fonksiyonu
            model_predictor: Model tahmin fonksiyonu
            validation_type: Validasyon tipi ('holdout', 'crossval', 'both')
            test_size: Hold-out için test seti oranı
            n_splits: Cross-validation için fold sayısı
            
        Returns:
            Dict[str, Dict]: Validasyon sonuçları
        """
        results = {}
        
        if validation_type in ['holdout', 'both']:
            try:
                results['hold_out'] = self.hold_out_validation(
                    data, model_trainer, model_predictor, test_size
                )
            except Exception as e:
                logger.error(f"Hold-out validasyonu başarısız: {e}")
                results['hold_out'] = {'error': str(e)}
        
        if validation_type in ['crossval', 'both']:
            try:
                results['cross_validation'] = self.time_series_cross_validation(
                    data, model_trainer, model_predictor, n_splits
                )
            except Exception as e:
                logger.error(f"Cross-validation başarısız: {e}")
                results['cross_validation'] = {'error': str(e)}
        
        return results


def validate_arima_model(data: pd.Series, forecaster_class, test_size: float = 0.2,
                         n_splits: int = 5) -> Dict[str, Dict]:
    """
    ARIMA modelini validate etmek için kolay kullanım fonksiyonu.
    
    Args:
        data: Zaman serisi verisi
        forecaster_class: ARIMAForecaster sınıfı
        test_size: Hold-out için test seti oranı
        n_splits: Cross-validation için fold sayısı
        
    Returns:
        Dict[str, Dict]: Validasyon sonuçları
    """
    def trainer(train_data):
        forecaster = forecaster_class()
        forecaster.fit(train_data, test_size=0.2)
        return forecaster
    
    def predictor(model, train_data, steps):
        results = model.forecast(steps=steps)
        return results['forecast']
    
    validator = ModelValidator()
    return validator.validate_model(data, trainer, predictor, 'both', test_size, n_splits)


if __name__ == '__main__':
    # Test
    logger.info("ModelValidator test ediliyor...")
    
    # Örnek veri
    dates = pd.date_range('2024-01-01', periods=100, freq='MS')
    data = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)
    
    # Basit model trainer ve predictor
    def simple_trainer(train_data):
        class SimpleModel:
            def predict(self, steps):
                return np.random.randn(steps).cumsum() + train_data.iloc[-1]
        return SimpleModel()
    
    def simple_predictor(model, train_data, steps):
        return model.predict(steps)
    
    validator = ModelValidator()
    results = validator.validate_model(data, simple_trainer, simple_predictor, 'both')
    
    logger.info(f"Validasyon sonuçları: {results}")
    logger.info("ModelValidator test tamamlandı.")

