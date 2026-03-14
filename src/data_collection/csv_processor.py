"""
CSV Formatı Otomatik İşleme Modülü
===================================

TÜBİTAK Projesi - CSV formatındaki verileri otomatik işleme

G-3 Gereksinimi: Sistem, CSV formatındaki verileri otomatik olarak işleyebilmelidir.

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
from typing import Optional, Dict, List, Union
from pathlib import Path
from datetime import datetime

# chardet import - encoding tespiti için
try:
    import chardet  # type: ignore
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
    chardet = None  # type: ignore
    logging.warning("chardet paketi yüklü değil. Encoding tespiti sınırlı olacak.")

# Proje kök dizinini import yolu olarak ekle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSVProcessor:
    """
    CSV formatındaki verileri otomatik olarak işleyen sınıf.
    
    Özellikler:
    - Otomatik encoding tespiti
    - Farklı delimiter desteği (virgül, noktalı virgül, tab)
    - Tarih formatı otomatik tanıma
    - Eksik değer işleme
    - Veri tipi otomatik tespiti
    """
    
    def __init__(self, encoding: Optional[str] = None, delimiter: Optional[str] = None):
        """
        CSVProcessor'ı başlatır.
        
        Args:
            encoding: Dosya encoding'i (None ise otomatik tespit)
            delimiter: CSV delimiter (None ise otomatik tespit)
        """
        self.encoding = encoding
        self.delimiter = delimiter
        logger.info("CSVProcessor başlatıldı.")
    
    def detect_encoding(self, file_path: Union[str, Path]) -> str:
        """
        CSV dosyasının encoding'ini tespit eder.
        
        Args:
            file_path: CSV dosya yolu
            
        Returns:
            str: Tespit edilen encoding
        """
        if not HAS_CHARDET:
            logger.warning("chardet paketi yüklü değil, UTF-8 kullanılıyor")
            return 'utf-8'
        
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # İlk 10KB'ı oku
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                logger.info(f"Encoding tespit edildi: {encoding} (güven: {result['confidence']:.2%})")
                return encoding if encoding else 'utf-8'
        except Exception as e:
            logger.warning(f"Encoding tespiti başarısız, UTF-8 kullanılıyor: {e}")
            return 'utf-8'
    
    def detect_delimiter(self, file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
        """
        CSV dosyasının delimiter'ını tespit eder.
        
        Args:
            file_path: CSV dosya yolu
            encoding: Dosya encoding'i
            
        Returns:
            str: Tespit edilen delimiter
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                first_line = f.readline()
                
            # Yaygın delimiter'ları kontrol et
            delimiters = [',', ';', '\t', '|']
            delimiter_counts = {d: first_line.count(d) for d in delimiters}
            
            if delimiter_counts:
                detected = max(delimiter_counts, key=delimiter_counts.get)
                if delimiter_counts[detected] > 0:
                    logger.info(f"Delimiter tespit edildi: '{detected}'")
                    return detected
            
            logger.warning("Delimiter tespit edilemedi, virgül kullanılıyor")
            return ','
        except Exception as e:
            logger.warning(f"Delimiter tespiti başarısız, virgül kullanılıyor: {e}")
            return ','
    
    def read_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        CSV dosyasını okur ve DataFrame'e dönüştürür.
        
        Args:
            file_path: CSV dosya yolu
            **kwargs: pandas.read_csv() için ek parametreler
            
        Returns:
            pd.DataFrame: Okunan veri
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV dosyası bulunamadı: {file_path}")
        
        # Encoding tespiti
        encoding = self.encoding or self.detect_encoding(file_path)
        
        # Delimiter tespiti
        delimiter = self.delimiter or self.detect_delimiter(file_path, encoding)
        
        # Okuma parametreleri
        read_params = {
            'filepath_or_buffer': file_path,
            'encoding': encoding,
            'delimiter': delimiter,
            'low_memory': False,
            'na_values': ['', 'NA', 'N/A', 'null', 'NULL', 'None', '-', '--'],
            **kwargs
        }
        
        try:
            df = pd.read_csv(**read_params)
            logger.info(f"CSV dosyası başarıyla okundu: {len(df)} satır, {len(df.columns)} sütun")
            return df
        except Exception as e:
            logger.error(f"CSV okuma hatası: {e}")
            raise
    
    def write_csv(self, df: pd.DataFrame, file_path: Union[str, Path], 
                  encoding: str = 'utf-8-sig', index: bool = False, **kwargs) -> bool:
        """
        DataFrame'i CSV dosyasına yazar.
        
        Args:
            df: Yazılacak DataFrame
            file_path: CSV dosya yolu
            encoding: Dosya encoding'i (varsayılan: utf-8-sig, BOM ile)
            index: İndeks yazılsın mı
            **kwargs: pandas.to_csv() için ek parametreler
            
        Returns:
            bool: Başarılı ise True
        """
        file_path = Path(file_path)
        
        try:
            # Klasör yoksa oluştur
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            write_params = {
                'path_or_buf': file_path,
                'encoding': encoding,
                'index': index,
                'na_rep': '',
                **kwargs
            }
            
            df.to_csv(**write_params)
            logger.info(f"CSV dosyası başarıyla yazıldı: {file_path}")
            return True
        except Exception as e:
            logger.error(f"CSV yazma hatası: {e}")
            return False
    
    def process_economic_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Ekonomik veri CSV'sini işler ve standart formata dönüştürür.
        
        Args:
            file_path: CSV dosya yolu
            
        Returns:
            pd.DataFrame: İşlenmiş veri
        """
        df = self.read_csv(file_path)
        
        # Tarih sütununu tespit et ve dönüştür
        date_columns = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['tarih', 'date', 'tarih', 'zaman', 'time'])]
        
        if date_columns:
            date_col = date_columns[0]
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.rename(columns={date_col: 'date'})
                df = df.set_index('date')
            except Exception as e:
                logger.warning(f"Tarih dönüştürme hatası: {e}")
        
        # Sayısal sütunları temizle
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Ekonomik veri işlendi: {len(df)} satır")
        return df
    
    def export_to_csv(self, data: Union[pd.DataFrame, Dict], 
                     file_path: Union[str, Path], **kwargs) -> bool:
        """
        Veriyi CSV formatına export eder.
        
        Args:
            data: Export edilecek veri (DataFrame veya Dict)
            file_path: Hedef CSV dosya yolu
            **kwargs: Ek parametreler
            
        Returns:
            bool: Başarılı ise True
        """
        if isinstance(data, dict):
            # Dict'i DataFrame'e dönüştür
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError(f"Desteklenmeyen veri tipi: {type(data)}")
        
        return self.write_csv(df, file_path, **kwargs)


def process_csv_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    CSV dosyasını işlemek için kolay kullanım fonksiyonu.
    
    Args:
        file_path: CSV dosya yolu
        
    Returns:
        pd.DataFrame: İşlenmiş veri
    """
    processor = CSVProcessor()
    return processor.process_economic_data(file_path)


def export_dataframe_to_csv(df: pd.DataFrame, file_path: Union[str, Path]) -> bool:
    """
    DataFrame'i CSV'ye export etmek için kolay kullanım fonksiyonu.
    
    Args:
        df: Export edilecek DataFrame
        file_path: Hedef CSV dosya yolu
        
    Returns:
        bool: Başarılı ise True
    """
    processor = CSVProcessor()
    return processor.export_to_csv(df, file_path)


if __name__ == '__main__':
    # Test
    logger.info("CSVProcessor test ediliyor...")
    processor = CSVProcessor()
    logger.info("CSVProcessor başarıyla başlatıldı.")

