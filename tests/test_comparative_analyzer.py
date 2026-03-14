"""
ComparativeAnalyzer Testleri
============================

TÜBİTAK Projesi - Karşılaştırmalı analiz için unit testler

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

from src.analysis.comparative_analyzer import ComparativeAnalyzer


class TestComparativeAnalyzer:
    """ComparativeAnalyzer için test sınıfı."""
    
    @pytest.fixture
    def sample_indicators(self):
        """Örnek gösterge verileri."""
        dates = pd.date_range('2024-01-01', periods=100, freq='MS')
        return {
            'usd_try': pd.Series(np.random.randn(100).cumsum() + 30, index=dates),
            'inflation': pd.Series(np.random.randn(100).cumsum() + 50, index=dates),
            'interest_rate': pd.Series(np.random.randn(100).cumsum() + 25, index=dates)
        }
    
    def test_calculate_correlation(self, sample_indicators):
        """Korelasyon hesaplama testi (G-12 Metrik 1)."""
        analyzer = ComparativeAnalyzer()
        result = analyzer.calculate_correlation(
            sample_indicators['usd_try'],
            sample_indicators['inflation']
        )
        
        assert 'correlation' in result
        assert -1 <= result['correlation'] <= 1
    
    def test_calculate_trend_similarity(self, sample_indicators):
        """Trend benzerliği hesaplama testi (G-12 Metrik 2)."""
        analyzer = ComparativeAnalyzer()
        result = analyzer.calculate_trend_similarity(
            sample_indicators['usd_try'],
            sample_indicators['inflation']
        )
        
        assert 'trend_similarity' in result
        assert 0 <= result['trend_similarity'] <= 1
    
    def test_calculate_volatility_comparison(self, sample_indicators):
        """Volatilite karşılaştırma testi (G-12 Metrik 3)."""
        analyzer = ComparativeAnalyzer()
        result = analyzer.calculate_volatility_comparison(
            sample_indicators['usd_try'],
            sample_indicators['inflation']
        )
        
        assert 'volatility_ratio' in result
        assert result['volatility_ratio'] >= 0
    
    def test_compare_indicators(self, sample_indicators):
        """Gösterge karşılaştırma testi (G-11)."""
        analyzer = ComparativeAnalyzer()
        results = analyzer.compare_indicators(sample_indicators)
        
        assert len(results) > 0
        # En az 3 karşılaştırma metriği olmalı (G-12)
        for comparison_key, metrics in results.items():
            assert 'correlation' in metrics
            assert 'trend_similarity' in metrics
            assert 'volatility_comparison' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

