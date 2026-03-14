"""
Türkiye'ye Özgü Veri Format Tanıma ve Standartlaştırma
=======================================================

TÜBİTAK Projesi - Türkiye'ye özgü veri formatlarını otomatik tanıma ve standartlaştırma

G-4 Gereksinimi: Sistem, Türkiye'ye özgü veri formatlarını (örn: ondalık ayıracı olarak 
virgül/nokta kullanımı, farklı tarih formatları) otomatik olarak tanımalı ve standart hale getirebilmelidir.

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Optional, Union, List, Tuple
from datetime import datetime
import locale

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TurkishFormatParser:
    """
    Türkiye'ye özgü veri formatlarını otomatik tanıyan ve standartlaştıran sınıf.
    
    Özellikler:
    - Ondalık ayıracı tanıma (virgül/nokta)
    - Binlik ayıracı tanıma (nokta/virgül)
    - Tarih formatı tanıma (DD.MM.YYYY, DD/MM/YYYY, MM-YYYY, vb.)
    - Türkçe karakter desteği
    - Sayısal değer standartlaştırma
    """
    
    # Türkiye'de yaygın tarih formatları
    DATE_FORMATS = [
        '%d.%m.%Y',      # 15.03.2024
        '%d/%m/%Y',      # 15/03/2024
        '%d-%m-%Y',      # 15-03-2024
        '%Y.%m.%d',      # 2024.03.15
        '%Y/%m/%d',      # 2024/03/15
        '%Y-%m-%d',      # 2024-03-15
        '%m-%Y',         # 03-2024
        '%m.%Y',         # 03.2024
        '%m/%Y',         # 03/2024
        '%Y-%m',         # 2024-03
        '%d.%m.%y',      # 15.03.24
        '%d/%m/%y',      # 15/03/24
    ]
    
    def __init__(self):
        """TurkishFormatParser'ı başlatır."""
        logger.info("TurkishFormatParser başlatıldı.")
    
    def detect_decimal_separator(self, value: str) -> Tuple[str, str]:
        """
        Ondalık ve binlik ayıracı tespit eder.
        
        Args:
            value: İncelenecek string değer
            
        Returns:
            Tuple[str, str]: (ondalık_ayıracı, binlik_ayıracı)
        """
        if not isinstance(value, str):
            value = str(value)
        
        # Virgül ve nokta sayısını say
        comma_count = value.count(',')
        dot_count = value.count('.')
        
        # Türkiye'de genellikle:
        # - Binlik ayıracı: nokta (1.234,56)
        # - Ondalık ayıracı: virgül (1.234,56)
        # VEYA
        # - Binlik ayıracı: virgül (1,234.56) - İngilizce format
        # - Ondalık ayıracı: nokta (1,234.56)
        
        if comma_count > 0 and dot_count > 0:
            # Her ikisi de varsa, son ayıracı ondalık ayıracı olarak kabul et
            last_comma = value.rfind(',')
            last_dot = value.rfind('.')
            
            if last_comma > last_dot:
                # Son virgül daha sonra: 1.234,56 formatı (Türk formatı)
                return (',', '.')
            else:
                # Son nokta daha sonra: 1,234.56 formatı (İngiliz formatı)
                return ('.', ',')
        elif comma_count > 0:
            # Sadece virgül varsa, muhtemelen Türk formatı (ondalık ayıracı)
            # Ama binlik ayıracı olarak da kullanılabilir
            # Sayı uzunluğuna göre karar ver
            parts = value.split(',')
            if len(parts) == 2 and len(parts[1]) <= 2:
                # Virgülden sonra 1-2 karakter: muhtemelen ondalık ayıracı
                return (',', '.')
            else:
                # Virgül binlik ayıracı olabilir
                return ('.', ',')
        elif dot_count > 0:
            # Sadece nokta varsa
            parts = value.split('.')
            if len(parts) == 2 and len(parts[1]) <= 2:
                # Noktadan sonra 1-2 karakter: muhtemelen ondalık ayıracı
                return ('.', ',')
            else:
                # Nokta binlik ayıracı olabilir
                return (',', '.')
        else:
            # Ayıracı yok
            return ('.', ',')  # Varsayılan: İngiliz formatı
    
    def parse_number(self, value: Union[str, float, int]) -> float:
        """
        Türkiye formatındaki sayıyı float'a çevirir.
        
        Args:
            value: Parse edilecek değer
            
        Returns:
            float: Parse edilmiş sayı
        """
        if pd.isna(value) or value == '' or value is None:
            return np.nan
        
        if isinstance(value, (int, float)):
            return float(value)
        
        value_str = str(value).strip()
        
        # Boş string kontrolü
        if not value_str or value_str == '-':
            return np.nan
        
        # Yüzde işareti varsa kaldır
        is_percentage = '%' in value_str
        value_str = value_str.replace('%', '').strip()
        
        # Ondalık ve binlik ayıracı tespit et
        decimal_sep, thousand_sep = self.detect_decimal_separator(value_str)
        
        # Binlik ayıracını kaldır
        if thousand_sep:
            value_str = value_str.replace(thousand_sep, '')
        
        # Ondalık ayıracını noktaya çevir
        if decimal_sep == ',':
            value_str = value_str.replace(',', '.')
        
        try:
            result = float(value_str)
            if is_percentage:
                result = result / 100.0
            return result
        except ValueError as e:
            logger.warning(f"Sayı parse edilemedi: {value_str} - {e}")
            return np.nan
    
    def parse_date(self, value: Union[str, datetime, pd.Timestamp]) -> Optional[pd.Timestamp]:
        """
        Türkiye formatındaki tarihi parse eder.
        
        Args:
            value: Parse edilecek tarih değeri
            
        Returns:
            pd.Timestamp: Parse edilmiş tarih veya None
        """
        if pd.isna(value) or value == '' or value is None:
            return None
        
        if isinstance(value, (datetime, pd.Timestamp)):
            return pd.Timestamp(value)
        
        value_str = str(value).strip()
        
        # Yaygın formatları dene
        for fmt in self.DATE_FORMATS:
            try:
                parsed = datetime.strptime(value_str, fmt)
                return pd.Timestamp(parsed)
            except ValueError:
                continue
        
        # pandas'ın otomatik parse'ını dene
        try:
            return pd.to_datetime(value_str, errors='coerce')
        except Exception as e:
            logger.warning(f"Tarih parse edilemedi: {value_str} - {e}")
            return None
    
    def standardize_dataframe(self, df: pd.DataFrame, 
                             date_columns: Optional[List[str]] = None,
                             numeric_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        DataFrame'deki Türkiye formatındaki verileri standartlaştırır.
        
        Args:
            df: Standartlaştırılacak DataFrame
            date_columns: Tarih sütunları listesi (None ise otomatik tespit)
            numeric_columns: Sayısal sütunlar listesi (None ise otomatik tespit)
            
        Returns:
            pd.DataFrame: Standartlaştırılmış DataFrame
        """
        df = df.copy()
        
        # Tarih sütunlarını tespit et
        if date_columns is None:
            date_columns = [col for col in df.columns if any(keyword in col.lower() 
                           for keyword in ['tarih', 'date', 'zaman', 'time'])]
        
        # Tarih sütunlarını parse et
        for col in date_columns:
            if col in df.columns:
                df[col] = df[col].apply(self.parse_date)
        
        # Sayısal sütunları tespit et
        if numeric_columns is None:
            # Tarih ve string sütunlarını hariç tut
            exclude_cols = set(date_columns)
            numeric_columns = [col for col in df.columns 
                            if col not in exclude_cols 
                            and df[col].dtype == 'object']
        
        # Sayısal sütunları parse et
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(self.parse_number)
        
        logger.info(f"DataFrame standartlaştırıldı: {len(date_columns)} tarih, {len(numeric_columns)} sayısal sütun")
        return df
    
    def detect_turkish_locale(self, sample_values: List[str]) -> bool:
        """
        Veri setinin Türk formatında olup olmadığını tespit eder.
        
        Args:
            sample_values: Örnek değerler listesi
            
        Returns:
            bool: Türk formatında ise True
        """
        if not sample_values:
            return False
        
        turkish_patterns = 0
        total_patterns = 0
        
        for value in sample_values[:10]:  # İlk 10 değeri kontrol et
            if not isinstance(value, str):
                continue
            
            total_patterns += 1
            
            # Türk formatı işaretleri:
            # - Virgül ondalık ayıracı olarak kullanılıyor (1.234,56)
            # - Nokta binlik ayıracı olarak kullanılıyor
            if ',' in value and '.' in value:
                # Son ayıracı kontrol et
                last_comma = value.rfind(',')
                last_dot = value.rfind('.')
                if last_comma > last_dot:
                    turkish_patterns += 1
        
        if total_patterns == 0:
            return False
        
        # %50'den fazla Türk formatı ise True döndür
        return (turkish_patterns / total_patterns) > 0.5


def parse_turkish_number(value: Union[str, float, int]) -> float:
    """
    Türkiye formatındaki sayıyı parse etmek için kolay kullanım fonksiyonu.
    
    Args:
        value: Parse edilecek değer
        
    Returns:
        float: Parse edilmiş sayı
    """
    parser = TurkishFormatParser()
    return parser.parse_number(value)


def parse_turkish_date(value: Union[str, datetime, pd.Timestamp]) -> Optional[pd.Timestamp]:
    """
    Türkiye formatındaki tarihi parse etmek için kolay kullanım fonksiyonu.
    
    Args:
        value: Parse edilecek tarih
        
    Returns:
        pd.Timestamp: Parse edilmiş tarih
    """
    parser = TurkishFormatParser()
    return parser.parse_date(value)


def standardize_turkish_dataframe(df: pd.DataFrame, 
                                  date_columns: Optional[List[str]] = None,
                                  numeric_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    DataFrame'deki Türkiye formatındaki verileri standartlaştırmak için kolay kullanım fonksiyonu.
    
    Args:
        df: Standartlaştırılacak DataFrame
        date_columns: Tarih sütunları listesi
        numeric_columns: Sayısal sütunlar listesi
        
    Returns:
        pd.DataFrame: Standartlaştırılmış DataFrame
    """
    parser = TurkishFormatParser()
    return parser.standardize_dataframe(df, date_columns, numeric_columns)


if __name__ == '__main__':
    # Test
    logger.info("TurkishFormatParser test ediliyor...")
    parser = TurkishFormatParser()
    
    # Test sayıları
    test_numbers = ["1.234,56", "1,234.56", "1234,56", "1234.56", "1.234"]
    for num in test_numbers:
        parsed = parser.parse_number(num)
        logger.info(f"{num} -> {parsed}")
    
    # Test tarihleri
    test_dates = ["15.03.2024", "15/03/2024", "03-2024", "2024-03-15"]
    for date in test_dates:
        parsed = parser.parse_date(date)
        logger.info(f"{date} -> {parsed}")
    
    logger.info("TurkishFormatParser test tamamlandı.")

