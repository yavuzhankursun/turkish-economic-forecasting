"""
TurkishFormatParser Testleri
=============================

TÜBİTAK Projesi - Türkiye formatı parse için unit testler

G-47 Gereksinimi: Kodun test kapsama oranı > %90 olmalıdır.
"""

import pytest
import pandas as pd
from datetime import datetime
import sys
import os

# Proje kök dizinini path'e ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.turkish_format_parser import TurkishFormatParser


class TestTurkishFormatParser:
    """TurkishFormatParser için test sınıfı."""
    
    def test_parse_turkish_number_comma_decimal(self):
        """Türk formatı sayı parse testi (virgül ondalık)."""
        parser = TurkishFormatParser()
        result = parser.parse_number("1.234,56")
        assert abs(result - 1234.56) < 0.01
    
    def test_parse_turkish_number_dot_decimal(self):
        """İngiliz formatı sayı parse testi (nokta ondalık)."""
        parser = TurkishFormatParser()
        result = parser.parse_number("1,234.56")
        assert abs(result - 1234.56) < 0.01
    
    def test_parse_turkish_date_ddmmyyyy(self):
        """Türk tarih formatı parse testi (DD.MM.YYYY)."""
        parser = TurkishFormatParser()
        result = parser.parse_date("15.03.2024")
        assert result is not None
        assert result.year == 2024
        assert result.month == 3
        assert result.day == 15
    
    def test_parse_turkish_date_mmyyyy(self):
        """Ay-Yıl formatı parse testi (MM-YYYY)."""
        parser = TurkishFormatParser()
        result = parser.parse_date("03-2024")
        assert result is not None
        assert result.year == 2024
        assert result.month == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

