"""
EconomicAnalyzer Testleri
=========================

TÜBİTAK Projesi - EconomicAnalyzer için unit testler

G-47 Gereksinimi: Kodun test kapsama oranı > %90 olmalıdır.
"""

import pytest
import sys
import os

# Proje kök dizinini path'e ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.economic_analyzer import EconomicAnalyzer


class TestEconomicAnalyzer:
    """EconomicAnalyzer için test sınıfı."""
    
    def test_initialization(self):
        """EconomicAnalyzer başlatma testi."""
        analyzer = EconomicAnalyzer()
        assert analyzer.usd_forecaster is None
        assert analyzer.inflation_forecaster is None
        assert analyzer.interest_forecaster is None
        assert analyzer.results == {}
    
    def test_generate_report_tr(self):
        """Türkçe rapor oluşturma testi (G-17)."""
        analyzer = EconomicAnalyzer()
        analyzer.results = {
            'timestamp': '2024-01-01T00:00:00',
            'usd_try': {
                'status': 'success',
                'last_value': 30.5,
                'data_points': 100,
                'model_params': (1, 1, 1),
                'metrics': {'RMSE': 0.5, 'MAE': 0.3, 'MAPE': 2.5}
            }
        }
        
        report = analyzer.generate_report(language='tr')
        assert report is not None
        assert 'USD/TRY' in report or 'KURU' in report
    
    def test_generate_report_en(self):
        """İngilizce rapor oluşturma testi (G-17)."""
        analyzer = EconomicAnalyzer()
        analyzer.results = {
            'timestamp': '2024-01-01T00:00:00',
            'usd_try': {
                'status': 'success',
                'last_value': 30.5,
                'data_points': 100,
                'model_params': (1, 1, 1),
                'metrics': {'RMSE': 0.5, 'MAE': 0.3, 'MAPE': 2.5}
            }
        }
        
        report = analyzer.generate_report(language='en')
        assert report is not None
        assert 'Exchange Rate' in report or 'USD/TRY' in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

