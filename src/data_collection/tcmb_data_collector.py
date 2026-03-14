"""
TCMB EVDS API - Gerçek Ekonomik Veri Toplama
=============================================

TÜBİTAK Projesi - TCMB'den enflasyon ve faiz oranı verisi çekme

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from datetime import datetime, timedelta
import sys
import os
import re

# Proje kök dizinini import yolu olarak ekle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.mongodb_manager import MongoDBManager

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TCMBDataCollector:
    """
    TCMB EVDS API'den gerçek ekonomik veri toplama.
    
    Veri kaynakları:
    - TCMB EVDS (Elektronik Veri Dağıtım Sistemi)
    - TÜFE (Enflasyon)
    - Politika Faiz Oranı
    """
    
    def __init__(self, api_key=None):
        """
        TCMB EVDS API key opsiyonel (public endpoint'ler için gerekmez)
        
        Args:
            api_key: TCMB EVDS API anahtarı (opsiyonel)
        """
        self.api_key = api_key
        self.base_url = "https://evds2.tcmb.gov.tr/service/evds/"
        logger.info("TCMB Data Collector başlatıldı.")
    
    def collect_inflation_data(self, years=3):
        """
        TCMB web sitesinden enflasyon (TÜFE) verisi web scraping ile çeker.
        
        Args:
            years: Kaç yıllık veri çekileceği
            
        Returns:
            pd.DataFrame: Tarih ve enflasyon oranı
        """
        logger.info(f"TCMB web sitesinden enflasyon verisi çekiliyor (web scraping)...")
        
        try:
            # TCMB resmi web sitesi
            url = "https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Istatistikler/Enflasyon+Verileri/Tuketici+Fiyatlari"
            
            logger.info(f"URL'ye bağlanılıyor: {url}")
            
            # HTTP isteği gönder
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            logger.info(f"Sayfa başarıyla indirildi. Status: {response.status_code}")
            
            # HTML parse et
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Tabloyu bul
            table = soup.find('table')
            
            if not table:
                logger.error("Tabloda veri bulunamadı. HTML yapısı değişmiş olabilir.")
                logger.error("❌ Mock data kullanımı yasak! Lütfen gerçek veri kaynağına erişimi kontrol edin.")
                raise ValueError("Enflasyon verisi çekilemedi. Mock data kullanımı yasak olduğu için işlem durduruldu.")
            
            # Tablo satırlarını parse et
            data_list = []
            rows = table.find_all('tr')[1:]  # İlk satır başlık
            
            logger.info(f"{len(rows)} satır veri bulundu.")
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    try:
                        # Tarih (MM-YYYY formatında)
                        date_str = cols[0].text.strip()
                        
                        # TÜFE Yıllık % Değişim
                        yearly_change_str = cols[1].text.strip()
                        
                        # Tarih parse et
                        date_obj = datetime.strptime(date_str, '%m-%Y')
                        
                        # Yüzde değerini temizle ve float'a çevir (Türk formatı desteği)
                        from src.utils.turkish_format_parser import parse_turkish_number
                        yearly_change = parse_turkish_number(yearly_change_str)
                        
                        data_list.append({
                            'date': date_obj,
                            'inflation_rate': yearly_change
                        })
                    except Exception as e:
                        logger.warning(f"Satır parse edilemedi: {e}")
                        continue
            
            if not data_list:
                logger.error("Hiç veri parse edilemedi.")
                logger.error("❌ Mock data kullanımı yasak! Lütfen gerçek veri kaynağına erişimi kontrol edin.")
                raise ValueError("Enflasyon verisi çekilemedi. Mock data kullanımı yasak olduğu için işlem durduruldu.")
            
            # DataFrame oluştur
            df = pd.DataFrame(data_list)
            df = df.sort_values('date')
            
            # Son N yıl filtrele
            cutoff_date = datetime.now() - timedelta(days=years*365)
            df = df[df['date'] >= cutoff_date]
            
            logger.info(f"✅ {len(df)} ay enflasyon verisi web scraping ile toplandı.")
            logger.info(f"Tarih aralığı: {df['date'].min()} - {df['date'].max()}")
            logger.info(f"Son değer: {df.iloc[-1]['inflation_rate']:.2f}%")
            
            return df
            
        except requests.RequestException as e:
            logger.error(f"TCMB web sitesine bağlanırken hata: {e}")
            logger.error("❌ Mock data kullanımı yasak! Lütfen gerçek veri kaynağına erişimi kontrol edin.")
            raise ValueError("Enflasyon verisi çekilemedi. Mock data kullanımı yasak olduğu için işlem durduruldu.")
        except Exception as e:
            logger.error(f"Web scraping hatası: {e}")
            logger.error("❌ Mock data kullanımı yasak! Lütfen gerçek veri kaynağına erişimi kontrol edin.")
            raise ValueError("Enflasyon verisi çekilemedi. Mock data kullanımı yasak olduğu için işlem durduruldu.")
    
    def collect_interest_rate_data(self, years=3):
        """
        TCMB web sitesinden politika faiz oranı verisi web scraping ile çeker.
        
        Args:
            years: Kaç yıllık veri çekileceği
            
        Returns:
            pd.DataFrame: Tarih ve faiz oranı
        """
        logger.info(f"TCMB web sitesinden faiz oranı verisi çekiliyor (web scraping)...")
        
        try:
            # TCMB resmi web sitesi - Merkez Bankası Faiz Oranları
            url = "https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Temel+Faaliyetler/Para+Politikasi/Merkez+Bankasi+Faiz+Oranlari/faiz-oranlari"
            
            logger.info(f"URL'ye bağlanılıyor: {url}")
            
            # HTTP isteği gönder
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            logger.info(f"Sayfa başarıyla indirildi. Status: {response.status_code}")
            
            # HTML parse et
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Tabloyu bul
            table = soup.find('table')
            
            if not table:
                logger.error("Tabloda veri bulunamadı. HTML yapısı değişmiş olabilir.")
                logger.error("❌ Mock data kullanımı yasak! Lütfen gerçek veri kaynağına erişimi kontrol edin.")
                raise ValueError("Faiz oranı verisi çekilemedi. Mock data kullanımı yasak olduğu için işlem durduruldu.")
            
            # Tablo satırlarını parse et
            data_list = []
            rows = table.find_all('tr')[1:]  # İlk satır başlık
            
            logger.info(f"{len(rows)} satır veri bulundu.")
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 3:  # Tarih, Borç Alma, Borç Verme
                    try:
                        # Tarih (DD.MM.YY formatında)
                        date_str = cols[0].text.strip()
                        
                        # Borç Verme oranı (3. sütun - politika faizi)
                        lending_rate_str = cols[2].text.strip()
                        
                        # Tarih parse et (DD.MM.YY)
                        date_obj = datetime.strptime(date_str, '%d.%m.%y')
                        
                        # Faiz oranını float'a çevir
                        lending_rate = float(lending_rate_str.replace(',', '.'))
                        
                        data_list.append({
                            'date': date_obj,
                            'policy_rate': lending_rate
                        })
                    except Exception as e:
                        logger.warning(f"Satır parse edilemedi: {e}")
                        continue
            
            if not data_list:
                logger.error("Hiç veri parse edilemedi.")
                logger.error("❌ Mock data kullanımı yasak! Lütfen gerçek veri kaynağına erişimi kontrol edin.")
                raise ValueError("Faiz oranı verisi çekilemedi. Mock data kullanımı yasak olduğu için işlem durduruldu.")
            
            # DataFrame oluştur
            df = pd.DataFrame(data_list)
            df = df.sort_values('date')
            
            # Son N yıl filtrele
            cutoff_date = datetime.now() - timedelta(days=years*365)
            df = df[df['date'] >= cutoff_date]
            
            # Aylık veri haline getir (her ayın son değerini al)
            df['year_month'] = df['date'].dt.to_period('M')
            df = df.groupby('year_month').last().reset_index()
            df['date'] = df['year_month'].dt.to_timestamp()
            df = df[['date', 'policy_rate']]
            
            logger.info(f"✅ {len(df)} ay faiz oranı verisi web scraping ile toplandı.")
            logger.info(f"Tarih aralığı: {df['date'].min()} - {df['date'].max()}")
            logger.info(f"Son değer: {df.iloc[-1]['policy_rate']:.2f}%")
            
            return df
            
        except requests.RequestException as e:
            logger.error(f"TCMB web sitesine bağlanırken hata: {e}")
            logger.error("❌ Mock data kullanımı yasak! Lütfen gerçek veri kaynağına erişimi kontrol edin.")
            raise ValueError("Faiz oranı verisi çekilemedi. Mock data kullanımı yasak olduğu için işlem durduruldu.")
        except Exception as e:
            logger.error(f"Web scraping hatası: {e}")
            logger.error("❌ Mock data kullanımı yasak! Lütfen gerçek veri kaynağına erişimi kontrol edin.")
            raise ValueError("Faiz oranı verisi çekilemedi. Mock data kullanımı yasak olduğu için işlem durduruldu.")
    
    def save_to_mongodb(self):
        """
        Toplanan verileri MongoDB'ye kaydeder.
        """
        logger.info("Veriler MongoDB'ye kaydediliyor...")
        
        try:
            with MongoDBManager() as mongodb_manager:
                # Enflasyon verisi
                inflation_df = self.collect_inflation_data()
                if inflation_df is not None and not inflation_df.empty:
                    saved_count = 0
                    for _, row in inflation_df.iterrows():
                        document = {
                            'data_type': 'economic_indicators',
                            'date': row['date'],
                            'inflation_rate': float(row['inflation_rate']),
                            'source': 'TÜİK',
                            'updated_at': datetime.now()
                        }
                        
                        # Upsert: aynı tarih varsa güncelle, yoksa ekle
                        collection = mongodb_manager.get_collection('economic_indicators')
                        collection.update_one(
                            {
                                'data_type': 'economic_indicators',
                                'date': row['date']
                            },
                            {'$set': document},
                            upsert=True
                        )
                        saved_count += 1
                    
                    logger.info(f"✓ {saved_count} enflasyon verisi kaydedildi.")
                
                # Faiz oranı verisi
                interest_df = self.collect_interest_rate_data()
                if interest_df is not None and not interest_df.empty:
                    saved_count = 0
                    for _, row in interest_df.iterrows():
                        # Aynı tarihi bul ve güncelle
                        collection = mongodb_manager.get_collection('economic_indicators')
                        collection.update_one(
                            {
                                'data_type': 'economic_indicators',
                                'date': row['date']
                            },
                            {
                                '$set': {
                                    'policy_rate': float(row['policy_rate']),
                                    'source': 'TCMB',
                                    'updated_at': datetime.now()
                                }
                            },
                            upsert=True
                        )
                        saved_count += 1
                    
                    logger.info(f"✓ {saved_count} faiz oranı verisi kaydedildi.")
                
                logger.info("Tüm veriler MongoDB'ye başarıyla kaydedildi!")
                return True
                
        except Exception as e:
            logger.error(f"MongoDB'ye kaydetme hatası: {e}")
            return False


def main():
    """Test ve veri toplama"""
    logger.info("=" * 60)
    logger.info("TCMB Gerçek Veri Toplama Başlatılıyor...")
    logger.info("=" * 60)
    
    collector = TCMBDataCollector()
    
    # Verileri topla ve MongoDB'ye kaydet
    success = collector.save_to_mongodb()
    
    if success:
        logger.info("\n✅ BAŞARILI! Gerçek TCMB/TÜİK verileri MongoDB'ye kaydedildi.")
        logger.info("Artık enflasyon ve faiz tahmini gerçek verilerle çalışacak.")
    else:
        logger.error("\n❌ HATA! Veriler kaydedilemedi.")
    
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

