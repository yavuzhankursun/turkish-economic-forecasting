"""
CSVProcessor Testleri
=====================

TÜBİTAK Projesi - CSV işleme için unit testler

G-47 Gereksinimi: Kodun test kapsama oranı > %90 olmalıdır.
"""

import pytest
import pandas as pd
import tempfile
import os
import sys

# Proje kök dizinini path'e ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_collection.csv_processor import CSVProcessor


class TestCSVProcessor:
    """CSVProcessor için test sınıfı."""
    
    @pytest.fixture
    def sample_csv_file(self):
        """Örnek CSV dosyası oluşturur."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'value': [1.5, 2.3, 3.1, 4.2, 5.0, 6.1, 7.2, 8.3, 9.4, 10.5]
        })
        
        # Geçici dosya oluştur
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        yield temp_file.name
        
        # Temizle
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
    
    def test_read_csv(self, sample_csv_file):
        """CSV okuma testi."""
        processor = CSVProcessor()
        df = processor.read_csv(sample_csv_file)
        
        assert df is not None
        assert len(df) > 0
        assert 'date' in df.columns or 'value' in df.columns
    
    def test_write_csv(self):
        """CSV yazma testi."""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        })
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.close()
        
        try:
            processor = CSVProcessor()
            success = processor.write_csv(df, temp_file.name)
            
            assert success is True
            assert os.path.exists(temp_file.name)
        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

