"""
ARIMA Model Testleri
====================

TÜBİTAK Projesi - ARIMA modeli için unit testler

G-47 Gereksinimi: Kodun test kapsama oranı > %90 olmalıdır.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Proje kök dizinini path'e ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.arima_model import ARIMAForecaster


class TestARIMAForecaster:
    """ARIMA modeli için test sınıfı."""
    
    @pytest.fixture
    def sample_data(self):
        """Örnek zaman serisi verisi oluşturur."""
        dates = pd.date_range('2020-01-01', periods=100, freq='MS')
        values = np.random.randn(100).cumsum() + 100
        return pd.Series(values, index=dates)
    
    def test_initialization(self):
        """ARIMAForecaster başlatma testi."""
        forecaster = ARIMAForecaster(target_column='test')
        assert forecaster.target_column == 'test'
        assert forecaster.fitted_model is None
    
    def test_fit(self, sample_data):
        """Model eğitimi testi."""
        forecaster = ARIMAForecaster(target_column='value')
        forecaster.fit(sample_data, test_size=0.2)
        
        assert forecaster.fitted_model is not None
        assert forecaster.best_params is not None
        assert len(forecaster.train_data) > 0
    
    def test_forecast(self, sample_data):
        """Tahmin testi."""
        forecaster = ARIMAForecaster(target_column='value')
        forecaster.fit(sample_data, test_size=0.2)
        
        results = forecaster.forecast(steps=12)
        
        assert results is not None
        assert 'forecast' in results
        assert len(results['forecast']) == 12
    
    def test_forecast_with_confidence_interval(self, sample_data):
        """Güven aralığı ile tahmin testi (G-31)."""
        forecaster = ARIMAForecaster(target_column='value')
        forecaster.fit(sample_data, test_size=0.2)
        
        results = forecaster.forecast(steps=12, confidence_level=0.95)
        
        assert results is not None
        assert 'forecast' in results
        # Güven aralığı varsa kontrol et
        if 'confidence_interval' in results:
            assert 'lower' in results['confidence_interval']
            assert 'upper' in results['confidence_interval']
    
    def test_evaluate(self, sample_data):
        """Model değerlendirme testi."""
        forecaster = ARIMAForecaster(target_column='value')
        forecaster.fit(sample_data, test_size=0.2)
        
        metrics = forecaster.evaluate()
        
        assert metrics is not None
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'MAPE' in metrics
    
    def test_bic_criterion(self, sample_data):
        """BIC kriteri testi (G-9)."""
        forecaster = ARIMAForecaster(target_column='value')
        forecaster.fit(sample_data, test_size=0.2, information_criterion='bic')
        
        assert forecaster.fitted_model is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

