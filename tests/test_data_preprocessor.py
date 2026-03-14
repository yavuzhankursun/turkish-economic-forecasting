"""
DataPreprocessor Testleri
==========================

TÜBİTAK Projesi - Veri ön işleme için unit testler

G-47 Gereksinimi: Kodun test kapsama oranı > %90 olmalıdır.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Proje kök dizinini path'e ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.data_preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """DataPreprocessor için test sınıfı."""
    
    @pytest.fixture
    def sample_data_with_missing(self):
        """Eksik değerler içeren örnek veri."""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        values = np.random.randn(50).cumsum() + 100
        df = pd.DataFrame({'value': values}, index=dates)
        # Bazı değerleri eksik yap
        df.loc[df.index[10:15], 'value'] = np.nan
        return df
    
    def test_detect_missing_values(self, sample_data_with_missing):
        """Eksik değer tespiti testi."""
        preprocessor = DataPreprocessor()
        missing = preprocessor.detect_missing_values(sample_data_with_missing)
        
        assert 'value' in missing
        assert missing['value'] > 0
    
    def test_interpolate_missing_values(self, sample_data_with_missing):
        """Eksik değer interpolasyonu testi."""
        preprocessor = DataPreprocessor()
        interpolated = preprocessor.interpolate_missing_values(
            sample_data_with_missing, method='time'
        )
        
        assert interpolated['value'].isnull().sum() < sample_data_with_missing['value'].isnull().sum()
    
    def test_detect_outliers(self):
        """Aykırı değer tespiti testi."""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        values = np.random.randn(50).cumsum() + 100
        # Aykırı değer ekle
        values[25] = 1000
        df = pd.DataFrame({'value': values}, index=dates)
        
        preprocessor = DataPreprocessor()
        outliers = preprocessor.detect_outliers(df, method='iqr')
        
        assert 'value' in outliers
        assert len(outliers['value']) > 0
    
    def test_convert_to_time_series(self):
        """Zaman serisi formatına dönüştürme testi."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50, freq='D'),
            'value': np.random.randn(50)
        })
        
        preprocessor = DataPreprocessor()
        ts_df = preprocessor.convert_to_time_series(df, date_column='date')
        
        assert isinstance(ts_df.index, pd.DatetimeIndex)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

