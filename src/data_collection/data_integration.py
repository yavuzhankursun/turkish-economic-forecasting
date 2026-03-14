"""
Farklı Kaynakları Birleştiren Entegrasyon Modülü
=================================================

TÜBİTAK Projesi - TÜİK, TCMB, BDDK verilerini tek formatta birleştirme

G-5 Gereksinimi: Sistem, farklı kaynaklardan (TÜİK, TCMB, BDDK) gelen verileri 
tek bir standardize formatta birleştirebilen bir entegrasyon modülüne sahip olmalıdır.

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import sys
import os

# Proje kök dizinini import yolu olarak ekle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.mongodb_manager import MongoDBManager
from src.utils.turkish_format_parser import TurkishFormatParser, standardize_turkish_dataframe
from src.data_collection.tcmb_data_collector import TCMBDataCollector
from src.data_collection.csv_processor import CSVProcessor

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIntegrationModule:
    """
    Farklı kaynaklardan gelen verileri tek formatta birleştiren entegrasyon modülü.
    
    Desteklenen kaynaklar:
    - TCMB (Türkiye Cumhuriyet Merkez Bankası)
    - TÜİK (Türkiye İstatistik Kurumu)
    - BDDK (Bankacılık Düzenleme ve Denetleme Kurumu)
    """
    
    # Standart sütun isimleri
    STANDARD_COLUMNS = {
        'date': 'date',
        'usd_try': 'usd_try',
        'eur_try': 'eur_try',
        'inflation_rate': 'inflation_rate',
        'policy_rate': 'policy_rate',
        'repo_rate': 'repo_rate',
        'lending_rate': 'lending_rate',
        'deposit_rate': 'deposit_rate',
    }
    
    def __init__(self):
        """DataIntegrationModule'ü başlatır."""
        self.tcmb_collector = TCMBDataCollector()
        self.csv_processor = CSVProcessor()
        self.format_parser = TurkishFormatParser()
        logger.info("DataIntegrationModule başlatıldı.")
    
    def standardize_column_names(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        DataFrame sütun isimlerini standart formata çevirir.
        
        Args:
            df: Standartlaştırılacak DataFrame
            source: Veri kaynağı (tcmb, tuik, bddk)
            
        Returns:
            pd.DataFrame: Standartlaştırılmış DataFrame
        """
        df = df.copy()
        
        # Kaynak bazlı sütun eşleştirmeleri
        column_mappings = {
            'tcmb': {
                'tarih': 'date',
                'usd': 'usd_try',
                'usd_try': 'usd_try',
                'eur': 'eur_try',
                'eur_try': 'eur_try',
                'enflasyon': 'inflation_rate',
                'inflation': 'inflation_rate',
                'inflation_rate': 'inflation_rate',
                'faiz': 'policy_rate',
                'policy_rate': 'policy_rate',
                'repo': 'repo_rate',
                'repo_rate': 'repo_rate',
            },
            'tuik': {
                'tarih': 'date',
                'date': 'date',
                'tufe': 'inflation_rate',
                'inflation': 'inflation_rate',
                'inflation_rate': 'inflation_rate',
                'enflasyon': 'inflation_rate',
            },
            'bddk': {
                'tarih': 'date',
                'date': 'date',
                'kredi_faizi': 'lending_rate',
                'lending_rate': 'lending_rate',
                'mevduat_faizi': 'deposit_rate',
                'deposit_rate': 'deposit_rate',
            }
        }
        
        mapping = column_mappings.get(source.lower(), {})
        
        # Sütun isimlerini eşleştir
        df = df.rename(columns=mapping)
        
        # Tarih sütununu standartlaştır
        if 'date' in df.columns:
            df['date'] = df['date'].apply(self.format_parser.parse_date)
            df = df.set_index('date')
        elif df.index.name in ['date', 'tarih'] or isinstance(df.index, pd.DatetimeIndex):
            df.index.name = 'date'
        
        return df
    
    def merge_dataframes(self, dataframes: List[pd.DataFrame], 
                        how: str = 'outer', sort: bool = True) -> pd.DataFrame:
        """
        Birden fazla DataFrame'i birleştirir.
        
        Args:
            dataframes: Birleştirilecek DataFrame listesi
            how: Birleştirme yöntemi ('outer', 'inner', 'left', 'right')
            sort: Sıralama yapılsın mı
            
        Returns:
            pd.DataFrame: Birleştirilmiş DataFrame
        """
        if not dataframes:
            return pd.DataFrame()
        
        # Tüm DataFrame'leri birleştir
        merged = dataframes[0]
        for df in dataframes[1:]:
            merged = merged.merge(df, left_index=True, right_index=True, how=how, suffixes=('', '_dup'))
            
            # Duplicate sütunları birleştir (aynı gösterge için)
            for col in merged.columns:
                if col.endswith('_dup'):
                    base_col = col[:-4]
                    if base_col in merged.columns:
                        # İki sütunu birleştir (NaN olmayan değerleri tercih et)
                        merged[base_col] = merged[base_col].fillna(merged[col])
                        merged = merged.drop(columns=[col])
        
        if sort:
            merged = merged.sort_index()
        
        logger.info(f"DataFrame'ler birleştirildi: {len(merged)} satır, {len(merged.columns)} sütun")
        return merged
    
    def integrate_tcmb_data(self, years: int = 3) -> pd.DataFrame:
        """
        TCMB verilerini entegre eder.
        
        Args:
            years: Kaç yıllık veri
            
        Returns:
            pd.DataFrame: TCMB verileri
        """
        try:
            logger.info("TCMB verileri entegre ediliyor...")
            
            # TCMB verilerini çek
            collector = TCMBDataCollector()
            collector.save_to_mongodb()
            
            # MongoDB'den oku
            db_manager = MongoDBManager()
            collection = db_manager.get_collection('economic_indicators')
            
            if collection is None:
                logger.error("MongoDB koleksiyonu bulunamadı")
                return pd.DataFrame()
            
            # Son N yıl veri çek
            cutoff_date = datetime.now() - timedelta(days=years*365)
            query = {'date': {'$gte': cutoff_date}}
            
            cursor = collection.find(query).sort('date', 1)
            documents = list(cursor)
            
            if not documents:
                logger.warning("TCMB verisi bulunamadı")
                return pd.DataFrame()
            
            # DataFrame'e dönüştür
            df = pd.DataFrame(documents)
            
            # Tarih sütununu parse et
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            
            # Standartlaştır
            df = self.standardize_column_names(df, 'tcmb')
            
            logger.info(f"TCMB verileri entegre edildi: {len(df)} satır")
            return df
            
        except Exception as e:
            logger.error(f"TCMB veri entegrasyonu hatası: {e}")
            return pd.DataFrame()
    
    def integrate_csv_data(self, file_path: Union[str, os.PathLike], 
                          source: str = 'unknown') -> pd.DataFrame:
        """
        CSV dosyasından veri entegre eder.
        
        Args:
            file_path: CSV dosya yolu
            source: Veri kaynağı (tcmb, tuik, bddk)
            
        Returns:
            pd.DataFrame: Entegre edilmiş veri
        """
        try:
            logger.info(f"CSV verileri entegre ediliyor: {file_path}")
            
            # CSV'yi oku
            df = self.csv_processor.read_csv(file_path)
            
            # Türk formatını standartlaştır
            df = standardize_turkish_dataframe(df)
            
            # Sütun isimlerini standartlaştır
            df = self.standardize_column_names(df, source)
            
            logger.info(f"CSV verileri entegre edildi: {len(df)} satır")
            return df
            
        except Exception as e:
            logger.error(f"CSV veri entegrasyonu hatası: {e}")
            return pd.DataFrame()
    
    def integrate_all_sources(self, years: int = 3, 
                            csv_files: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Tüm kaynaklardan verileri entegre eder.
        
        Args:
            years: Kaç yıllık veri
            csv_files: CSV dosya yolları dict'i {'source': 'path'}
            
        Returns:
            pd.DataFrame: Tüm kaynaklardan birleştirilmiş veri
        """
        logger.info("Tüm kaynaklardan veriler entegre ediliyor...")
        
        dataframes = []
        
        # TCMB verilerini ekle
        tcmb_df = self.integrate_tcmb_data(years)
        if not tcmb_df.empty:
            dataframes.append(tcmb_df)
        
        # CSV dosyalarını ekle
        if csv_files:
            for source, file_path in csv_files.items():
                csv_df = self.integrate_csv_data(file_path, source)
                if not csv_df.empty:
                    dataframes.append(csv_df)
        
        # Tüm DataFrame'leri birleştir
        if dataframes:
            merged_df = self.merge_dataframes(dataframes, how='outer')
            logger.info(f"Tüm kaynaklar entegre edildi: {len(merged_df)} satır, {len(merged_df.columns)} sütun")
            return merged_df
        else:
            logger.warning("Entegre edilecek veri bulunamadı")
            return pd.DataFrame()
    
    def save_integrated_data(self, df: pd.DataFrame, collection_name: str = 'integrated_economic_data') -> bool:
        """
        Entegre edilmiş veriyi MongoDB'ye kaydeder.
        
        Args:
            df: Kaydedilecek DataFrame
            collection_name: MongoDB koleksiyon adı
            
        Returns:
            bool: Başarılı ise True
        """
        try:
            db_manager = MongoDBManager()
            collection = db_manager.get_collection(collection_name)
            
            if collection is None:
                logger.error("MongoDB koleksiyonu bulunamadı")
                return False
            
            # DataFrame'i MongoDB formatına çevir
            documents = []
            for date, row in df.iterrows():
                doc = {
                    'date': pd.Timestamp(date) if isinstance(date, str) else date,
                    'created_at': datetime.now(),
                    'updated_at': datetime.now(),
                    'source': 'integrated'
                }
                
                # Sütunları ekle
                for col, value in row.items():
                    if pd.notna(value):
                        doc[col] = float(value) if isinstance(value, (int, float)) else str(value)
                
                documents.append(doc)
            
            # Eski verileri temizle ve yeni verileri ekle
            collection.delete_many({'source': 'integrated'})
            if documents:
                collection.insert_many(documents)
            
            logger.info(f"Entegre veriler MongoDB'ye kaydedildi: {len(documents)} belge")
            return True
            
        except Exception as e:
            logger.error(f"MongoDB kaydetme hatası: {e}")
            return False


def integrate_economic_data(years: int = 3, 
                           csv_files: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Tüm ekonomik verileri entegre etmek için kolay kullanım fonksiyonu.
    
    Args:
        years: Kaç yıllık veri
        csv_files: CSV dosya yolları dict'i
        
    Returns:
        pd.DataFrame: Entegre edilmiş veri
    """
    integrator = DataIntegrationModule()
    return integrator.integrate_all_sources(years, csv_files)


if __name__ == '__main__':
    # Test
    logger.info("DataIntegrationModule test ediliyor...")
    integrator = DataIntegrationModule()
    
    # TCMB verilerini entegre et
    df = integrator.integrate_tcmb_data(years=1)
    logger.info(f"Entegre edilen veri: {len(df)} satır")
    
    logger.info("DataIntegrationModule test tamamlandı.")

