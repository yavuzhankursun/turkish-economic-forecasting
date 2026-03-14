"""
Support Vector Regression (SVR) Modeli
======================================

TÜBİTAK Projesi - SVR modeli ile tahmin

G-10 Gereksinimi: Sistem, "hibrit bir metodoloji" (ARIMA-SVR) kullanmalıdır.

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Tuple
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SVRForecaster:
    """
    Support Vector Regression (SVR) tahmin modeli.
    
    ARIMA ile hibrit kullanım için tasarlanmıştır.
    """
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, epsilon: float = 0.1):
        """
        SVRForecaster'ı başlatır.
        
        Args:
            kernel: Kernel tipi ('rbf', 'linear', 'poly')
            C: Regularization parametresi
            epsilon: Epsilon-tube genişliği
        """
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        logger.info(f"SVRForecaster başlatıldı (kernel: {kernel}, C: {C}, epsilon: {epsilon})")
    
    def _create_features(self, data: pd.Series, lag: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zaman serisi verisinden özellikler oluşturur.
        
        Args:
            data: Zaman serisi verisi
            lag: Kaç lag kullanılacak
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y) özellik ve hedef matrisleri
        """
        values = data.values
        n = len(values)
        
        if n < lag + 1:
            raise ValueError(f"Yetersiz veri: {n} < {lag + 1}")
        
        X = []
        y = []
        
        for i in range(lag, n):
            X.append(values[i-lag:i])
            y.append(values[i])
        
        return np.array(X), np.array(y)
    
    def fit(self, data: pd.Series, lag: int = 5):
        """
        SVR modelini eğitir.
        
        Args:
            data: Eğitim verisi
            lag: Lag sayısı
        """
        if data is None or len(data) < lag + 1:
            raise ValueError(f"Yetersiz veri: {len(data)} < {lag + 1}")
        
        # Özellikler oluştur
        X, y = self._create_features(data, lag)
        
        # Normalleştir
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Modeli eğit
        self.model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
        self.model.fit(X_scaled, y_scaled)
        
        self.is_fitted = True
        self.lag = lag
        logger.info(f"SVR modeli eğitildi: {len(X)} örnek, {lag} lag")
    
    def predict(self, data: pd.Series, steps: int = 12) -> pd.Series:
        """
        Gelecek değerleri tahmin eder.
        
        Args:
            data: Tahmin için kullanılacak son veriler
            steps: Kaç adım ilerisi tahmin edilecek
            
        Returns:
            pd.Series: Tahmin edilen değerler
        """
        if not self.is_fitted:
            raise ValueError("Model henüz eğitilmedi!")
        
        if len(data) < self.lag:
            raise ValueError(f"Yetersiz veri: {len(data)} < {self.lag}")
        
        predictions = []
        current_data = data.values[-self.lag:].copy()
        
        for _ in range(steps):
            # Son lag değerlerinden özellik oluştur
            X = current_data[-self.lag:].reshape(1, -1)
            X_scaled = self.scaler_X.transform(X)
            
            # Tahmin yap
            y_pred_scaled = self.model.predict(X_scaled)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[0]
            
            predictions.append(y_pred)
            
            # Tahmin edilen değeri ekle ve bir sonraki tahmin için kullan
            current_data = np.append(current_data, y_pred)
        
        # Tarih indeksi oluştur
        if isinstance(data.index, pd.DatetimeIndex):
            last_date = data.index[-1]
            freq = pd.infer_freq(data.index) or 'MS'
            forecast_index = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]
        else:
            forecast_index = pd.RangeIndex(start=len(data), stop=len(data) + steps)
        
        return pd.Series(predictions, index=forecast_index)
    
    def evaluate(self, test_data: pd.Series) -> Dict[str, float]:
        """
        Model performansını değerlendirir.
        
        Args:
            test_data: Test verisi
            
        Returns:
            Dict[str, float]: Performans metrikleri
        """
        if not self.is_fitted:
            raise ValueError("Model henüz eğitilmedi!")
        
        if len(test_data) < self.lag + 1:
            raise ValueError(f"Yetersiz test verisi: {len(test_data)} < {self.lag + 1}")
        
        # Tahmin yap
        predictions = []
        actuals = []
        
        for i in range(self.lag, len(test_data)):
            window = test_data.iloc[i-self.lag:i]
            pred = self.predict(window, steps=1).iloc[0]
            predictions.append(pred)
            actuals.append(test_data.iloc[i])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Metrikleri hesapla
        mae = mean_absolute_error(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        
        # MAPE hesapla
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-10))) * 100
        
        metrics = {
            'MAE': float(mae),
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAPE': float(mape)
        }
        
        logger.info(f"SVR Performans: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
        return metrics


class ARIMASVRHybrid:
    """
    ARIMA-SVR Hibrit Modeli
    
    G-10 gereksinimi: ARIMA ve SVR modellerini birleştiren hibrit metodoloji.
    """
    
    def __init__(self, arima_model, svr_kernel: str = 'rbf', svr_C: float = 1.0, 
                 svr_epsilon: float = 0.1, weight_arima: float = 0.6, weight_svr: float = 0.4):
        """
        ARIMA-SVR Hibrit modelini başlatır.
        
        Args:
            arima_model: Eğitilmiş ARIMA modeli
            svr_kernel: SVR kernel tipi
            svr_C: SVR C parametresi
            svr_epsilon: SVR epsilon parametresi
            weight_arima: ARIMA ağırlığı
            weight_svr: SVR ağırlığı
        """
        self.arima_model = arima_model
        self.svr_model = SVRForecaster(kernel=svr_kernel, C=svr_C, epsilon=svr_epsilon)
        self.weight_arima = weight_arima
        self.weight_svr = weight_svr
        
        # Ağırlıkları normalize et
        total_weight = weight_arima + weight_svr
        if total_weight > 0:
            self.weight_arima = weight_arima / total_weight
            self.weight_svr = weight_svr / total_weight
        
        logger.info(f"ARIMA-SVR Hibrit modeli başlatıldı (ARIMA: {self.weight_arima:.2f}, SVR: {self.weight_svr:.2f})")
    
    def fit(self, data: pd.Series, svr_lag: int = 5):
        """
        Hibrit modeli eğitir.
        
        Args:
            data: Eğitim verisi
            svr_lag: SVR için lag sayısı
        """
        # ARIMA zaten eğitilmiş olmalı
        if self.arima_model.fitted_model is None:
            raise ValueError("ARIMA modeli henüz eğitilmedi!")
        
        # SVR'ı eğit
        self.svr_model.fit(data, lag=svr_lag)
        logger.info("ARIMA-SVR hibrit modeli eğitildi")
    
    def forecast(self, data: pd.Series, steps: int = 12) -> Dict:
        """
        Hibrit tahmin yapar.
        
        Args:
            data: Tahmin için kullanılacak veri
            steps: Kaç adım ilerisi tahmin edilecek
            
        Returns:
            Dict: Tahmin sonuçları
        """
        # ARIMA tahmini
        arima_results = self.arima_model.forecast(steps=steps)
        arima_forecast = arima_results['forecast']
        
        # SVR tahmini
        svr_forecast = self.svr_model.predict(data, steps=steps)
        
        # Hibrit tahmin (ağırlıklı ortalama)
        hybrid_forecast = (
            self.weight_arima * arima_forecast.values + 
            self.weight_svr * svr_forecast.values
        )
        
        # Tarih indeksi
        forecast_index = arima_forecast.index
        
        results = {
            'forecast': pd.Series(hybrid_forecast, index=forecast_index),
            'arima_forecast': arima_forecast,
            'svr_forecast': svr_forecast,
            'weights': {
                'arima': self.weight_arima,
                'svr': self.weight_svr
            }
        }
        
        # Güven aralığı varsa ekle
        if 'confidence_interval' in arima_results:
            results['confidence_interval'] = arima_results['confidence_interval']
        
        logger.info(f"ARIMA-SVR hibrit tahmin tamamlandı: {steps} adım")
        return results


if __name__ == '__main__':
    # Test
    logger.info("SVRForecaster test ediliyor...")
    
    # Örnek veri
    dates = pd.date_range('2024-01-01', periods=100, freq='MS')
    data = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)
    
    # SVR modeli
    svr = SVRForecaster()
    svr.fit(data, lag=5)
    predictions = svr.predict(data, steps=12)
    
    logger.info(f"SVR tahminleri: {len(predictions)} adım")
    logger.info("SVRForecaster test tamamlandı.")

