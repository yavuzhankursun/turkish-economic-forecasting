"""
Performans İzleme Modülü
========================

TÜBİTAK Projesi - Bellek kullanımı ve performans metrikleri

G-44 Gereksinimi: Sistemin bellek kullanımı < 64 MB altında tutulmalıdır.
G-45 Gereksinimi: Veri işleme doğruluğu > %99 olmalıdır.
G-46 Gereksinimi: Veri kaybı (işleme sırasında) < %1 olmalıdır.

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

import psutil
import os
import logging
import tracemalloc
from typing import Dict, Optional
import pandas as pd
import numpy as np
from functools import wraps
import time

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Performans izleme sınıfı.
    
    Özellikler:
    - Bellek kullanımı izleme
    - Veri işleme doğruluğu hesaplama
    - Veri kaybı tespiti
    """
    
    TARGET_MEMORY_MB = 64  # G-44: < 64 MB
    TARGET_PROCESSING_ACCURACY = 0.99  # G-45: > %99
    TARGET_DATA_LOSS_RATE = 0.01  # G-46: < %1
    
    def __init__(self):
        """PerformanceMonitor'ı başlatır."""
        self.process = psutil.Process(os.getpid())
        logger.info("PerformanceMonitor başlatıldı.")
    
    def get_memory_usage(self) -> Dict:
        """
        Bellek kullanımını ölçer.
        
        G-44 gereksinimi: < 64 MB
        
        Returns:
            Dict: Bellek kullanımı metrikleri
        """
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # MB cinsinden
        
        meets_target = memory_mb < self.TARGET_MEMORY_MB
        
        result = {
            'memory_mb': float(memory_mb),
            'target_mb': self.TARGET_MEMORY_MB,
            'meets_target': meets_target,
            'status': 'PASS' if meets_target else 'FAIL'
        }
        
        status_icon = '✅' if meets_target else '❌'
        logger.info(
            f"{status_icon} Bellek kullanımı: {memory_mb:.2f} MB "
            f"(Hedef: < {self.TARGET_MEMORY_MB} MB) - {'BAŞARILI' if meets_target else 'HEDEF AŞILDI'}"
        )
        
        return result
    
    def calculate_processing_accuracy(self, original_data: pd.DataFrame,
                                     processed_data: pd.DataFrame) -> Dict:
        """
        Veri işleme doğruluğunu hesaplar.
        
        G-45 gereksinimi: > %99
        
        Args:
            original_data: Orijinal veri
            processed_data: İşlenmiş veri
            
        Returns:
            Dict: İşleme doğruluğu metrikleri
        """
        # Ortak sütunları bul
        common_columns = set(original_data.columns) & set(processed_data.columns)
        
        if not common_columns:
            return {
                'accuracy': 0.0,
                'meets_target': False,
                'target': self.TARGET_PROCESSING_ACCURACY
            }
        
        total_cells = 0
        accurate_cells = 0
        
        for col in common_columns:
            orig_col = original_data[col].dropna()
            proc_col = processed_data[col].dropna()
            
            # Ortak indeksleri bul
            common_idx = orig_col.index.intersection(proc_col.index)
            
            if len(common_idx) == 0:
                continue
            
            orig_values = orig_col.loc[common_idx]
            proc_values = proc_col.loc[common_idx]
            
            # Sayısal sütunlar için doğruluk kontrolü
            if pd.api.types.is_numeric_dtype(orig_col):
                # Yüzde farkı %1'den azsa doğru kabul et
                diff_pct = np.abs((orig_values - proc_values) / (orig_values + 1e-10))
                accurate = (diff_pct < 0.01).sum()
                total_cells += len(common_idx)
                accurate_cells += accurate
            else:
                # Kategorik sütunlar için tam eşleşme
                matches = (orig_values == proc_values).sum()
                total_cells += len(common_idx)
                accurate_cells += matches
        
        if total_cells == 0:
            accuracy = 0.0
        else:
            accuracy = accurate_cells / total_cells
        
        meets_target = accuracy >= self.TARGET_PROCESSING_ACCURACY
        
        result = {
            'accuracy': float(accuracy),
            'accuracy_percentage': float(accuracy * 100),
            'target': self.TARGET_PROCESSING_ACCURACY,
            'target_percentage': self.TARGET_PROCESSING_ACCURACY * 100,
            'total_cells': total_cells,
            'accurate_cells': accurate_cells,
            'meets_target': meets_target,
            'status': 'PASS' if meets_target else 'FAIL'
        }
        
        status_icon = '✅' if meets_target else '❌'
        logger.info(
            f"{status_icon} Veri işleme doğruluğu: {accuracy*100:.2f}% "
            f"(Hedef: > {self.TARGET_PROCESSING_ACCURACY*100:.0f}%) - {'BAŞARILI' if meets_target else 'HEDEF AŞILDI'}"
        )
        
        return result
    
    def calculate_data_loss(self, original_data: pd.DataFrame,
                           processed_data: pd.DataFrame) -> Dict:
        """
        Veri kaybını hesaplar.
        
        G-46 gereksinimi: < %1
        
        Args:
            original_data: Orijinal veri
            processed_data: İşlenmiş veri
            
        Returns:
            Dict: Veri kaybı metrikleri
        """
        # Toplam veri noktası sayısı
        original_count = original_data.size
        processed_count = processed_data.size
        
        # Eksik değer sayısı
        original_missing = original_data.isnull().sum().sum()
        processed_missing = processed_data.isnull().sum().sum()
        
        # Veri kaybı = işleme sonrası eklenen eksik değerler
        additional_missing = processed_missing - original_missing
        data_loss_rate = additional_missing / (original_count + 1e-10)
        
        # Negatif olmamalı (eksik değerler azalabilir)
        data_loss_rate = max(0.0, data_loss_rate)
        
        meets_target = data_loss_rate < self.TARGET_DATA_LOSS_RATE
        
        result = {
            'data_loss_rate': float(data_loss_rate),
            'data_loss_percentage': float(data_loss_rate * 100),
            'target': self.TARGET_DATA_LOSS_RATE,
            'target_percentage': self.TARGET_DATA_LOSS_RATE * 100,
            'original_count': int(original_count),
            'processed_count': int(processed_count),
            'original_missing': int(original_missing),
            'processed_missing': int(processed_missing),
            'additional_missing': int(additional_missing),
            'meets_target': meets_target,
            'status': 'PASS' if meets_target else 'FAIL'
        }
        
        status_icon = '✅' if meets_target else '❌'
        logger.info(
            f"{status_icon} Veri kaybı: {data_loss_rate*100:.2f}% "
            f"(Hedef: < {self.TARGET_DATA_LOSS_RATE*100:.0f}%) - {'BAŞARILI' if meets_target else 'HEDEF AŞILDI'}"
        )
        
        return result
    
    def monitor_function(self, func):
        """
        Fonksiyon performansını izlemek için decorator.
        
        Args:
            func: İzlenecek fonksiyon
            
        Returns:
            Decorated function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Bellek izlemeyi başlat
            tracemalloc.start()
            start_memory = self.get_memory_usage()['memory_mb']
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Performans metrikleri
                end_time = time.time()
                end_memory = self.get_memory_usage()['memory_mb']
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                peak_memory_mb = peak / (1024 * 1024)
                
                logger.info(
                    f"Fonksiyon {func.__name__}: "
                    f"Süre={execution_time:.2f}s, "
                    f"Bellek={memory_delta:.2f} MB, "
                    f"Peak={peak_memory_mb:.2f} MB"
                )
                
                return result
            except Exception as e:
                tracemalloc.stop()
                logger.error(f"Fonksiyon {func.__name__} hatası: {e}")
                raise
        
        return wrapper


def monitor_performance(original_data: Optional[pd.DataFrame] = None,
                       processed_data: Optional[pd.DataFrame] = None) -> Dict:
    """
    Performans metriklerini hesaplamak için kolay kullanım fonksiyonu.
    
    Args:
        original_data: Orijinal veri (opsiyonel)
        processed_data: İşlenmiş veri (opsiyonel)
        
    Returns:
        Dict: Performans metrikleri
    """
    monitor = PerformanceMonitor()
    results = {
        'memory_usage': monitor.get_memory_usage()
    }
    
    if original_data is not None and processed_data is not None:
        results['processing_accuracy'] = monitor.calculate_processing_accuracy(
            original_data, processed_data
        )
        results['data_loss'] = monitor.calculate_data_loss(
            original_data, processed_data
        )
    
    return results


if __name__ == '__main__':
    # Test
    logger.info("PerformanceMonitor test ediliyor...")
    
    monitor = PerformanceMonitor()
    
    # Bellek kullanımı
    memory = monitor.get_memory_usage()
    logger.info(f"Bellek kullanımı: {memory}")
    
    # Veri işleme doğruluğu testi
    original = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
    processed = pd.DataFrame({'value': [1.001, 2.002, 3.0, 4.0, 5.0]})
    
    accuracy = monitor.calculate_processing_accuracy(original, processed)
    data_loss = monitor.calculate_data_loss(original, processed)
    
    logger.info(f"İşleme doğruluğu: {accuracy}")
    logger.info(f"Veri kaybı: {data_loss}")
    
    logger.info("PerformanceMonitor test tamamlandı.")

