"""
Günlük Veri Güncelleme Scheduler
=================================

TÜBİTAK Projesi - Otomatik günlük veri güncelleme

G-28 Gereksinimi: Veri güncelleme sıklığı "Günlük" olmalıdır.

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

import schedule
import time
import logging
import threading
from datetime import datetime
from typing import Optional, Callable
import sys
import os

# Proje kök dizinini import yolu olarak ekle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.data_collection.tcmb_data_collector import TCMBDataCollector

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataScheduler:
    """
    Günlük veri güncelleme scheduler'ı.
    
    Özellikler:
    - Günlük otomatik veri güncelleme
    - Özelleştirilebilir güncelleme zamanı
    - Thread-safe çalışma
    """
    
    def __init__(self, update_time: str = "09:00"):
        """
        DataScheduler'ı başlatır.
        
        Args:
            update_time: Güncelleme saati (HH:MM formatında)
        """
        self.update_time = update_time
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        logger.info(f"DataScheduler başlatıldı (güncelleme saati: {update_time})")
    
    def update_data(self):
        """
        Verileri günceller.
        
        G-28 gereksinimi: Günlük veri güncelleme
        """
        logger.info("=" * 80)
        logger.info("GÜNLÜK VERİ GÜNCELLEME BAŞLATILIYOR...")
        logger.info(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        try:
            # TCMB verilerini güncelle
            collector = TCMBDataCollector()
            success = collector.save_to_mongodb()
            
            if success:
                logger.info("✅ Günlük veri güncelleme başarılı")
            else:
                logger.error("❌ Günlük veri güncelleme başarısız")
                
        except Exception as e:
            logger.error(f"❌ Veri güncelleme hatası: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info("=" * 80)
    
    def start(self):
        """
        Scheduler'ı başlatır.
        """
        if self.is_running:
            logger.warning("Scheduler zaten çalışıyor")
            return
        
        # Günlük güncelleme zamanını ayarla
        schedule.every().day.at(self.update_time).do(self.update_data)
        
        self.is_running = True
        
        # Scheduler'ı ayrı thread'de çalıştır
        def run_scheduler():
            logger.info(f"Scheduler başlatıldı - Günlük güncelleme saati: {self.update_time}")
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Her dakika kontrol et
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("✅ DataScheduler başlatıldı")
    
    def stop(self):
        """
        Scheduler'ı durdurur.
        """
        self.is_running = False
        schedule.clear()
        logger.info("DataScheduler durduruldu")
    
    def run_now(self):
        """
        Hemen veri güncellemesi yapar (test için).
        """
        logger.info("Manuel veri güncelleme başlatılıyor...")
        self.update_data()


def start_daily_data_update(update_time: str = "09:00") -> DataScheduler:
    """
    Günlük veri güncelleme scheduler'ını başlatmak için kolay kullanım fonksiyonu.
    
    Args:
        update_time: Güncelleme saati (HH:MM formatında)
        
    Returns:
        DataScheduler: Scheduler instance'ı
    """
    scheduler = DataScheduler(update_time)
    scheduler.start()
    return scheduler


if __name__ == '__main__':
    # Test
    logger.info("DataScheduler test ediliyor...")
    
    scheduler = DataScheduler(update_time="09:00")
    
    # Hemen çalıştır (test için)
    scheduler.run_now()
    
    # Scheduler'ı başlat
    scheduler.start()
    
    logger.info("DataScheduler test tamamlandı. Scheduler çalışıyor...")
    logger.info("Çıkmak için Ctrl+C basın")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scheduler.stop()
        logger.info("Scheduler durduruldu")

