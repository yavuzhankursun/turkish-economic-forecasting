"""
Veritabanı Yönetim Modülü
=========================

Bu modül, PostgreSQL veritabanı ile olan tüm etkileşimleri yönetir.
SQLAlchemy kütüphanesi kullanılarak bağlantı kurulur ve veri işlemleri yapılır.

Proje: TÜBİTAK Ekonomik Göstergeler Tahmin Sistemi
Geliştirici: [Öğrenci Adı]
Tarih: 2024
"""

import os
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional
import sys
from pathlib import Path

# --- Proje Kök Dizinini Path'e Ekleme ---
sys.path.append(str(Path(__file__).resolve().parents[2]))
# -----------------------------------------

# Proje konfigürasyonunu import et
from config.config import DATABASE_CONFIG

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    PostgreSQL veritabanı işlemlerini yöneten sınıf.
    
    Özellikler:
    - Veritabanı bağlantısı kurma ve yönetme
    - Pandas DataFrame'lerini veritabanına kaydetme
    - Ham SQL sorguları çalıştırma
    - Bağlantı havuzu (connection pooling)
    """
    
    def __init__(self, env: str = 'production'):
        """
        DatabaseManager'ı başlatır ve veritabanı motorunu oluşturur.
        
        Args:
            env (str): Çalışma ortamı ('development' veya 'production').
                       Bu, config dosyasından hangi ayarların kullanılacağını belirler.
        """
        self.config = DATABASE_CONFIG.get(env)
        if not self.config:
            raise ValueError(f"'{env}' için veritabanı konfigürasyonu bulunamadı!")
            
        self.engine: Optional[Engine] = None
        self._create_engine()
        
    def _create_engine(self):
        """
        SQLAlchemy veritabanı motorunu (engine) oluşturur.
        Bu motor, veritabanı ile olan tüm iletişimi sağlar.
        """
        try:
            db_type = self.config['type']
            
            if db_type == 'postgresql':
                db_url = (
                    f"postgresql+psycopg2://{self.config['username']}:{self.config['password']}"
                    f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
                )
            elif db_type == 'sqlite':
                db_url = f"sqlite:///{self.config['path']}"
            else:
                raise ValueError(f"Desteklenmeyen veritabanı türü: {db_type}")

            logger.info(f"🔌 Veritabanı motoru oluşturuluyor: {db_type} @ {self.config.get('host', 'localhost')}")
            
            self.engine = create_engine(
                db_url,
                pool_size=self.config.get('pool_size', 5),
                max_overflow=self.config.get('max_overflow', 10),
                echo=self.config.get('echo', False)
            )
            
            # Bağlantıyı test et
            self.test_connection()

        except SQLAlchemyError as e:
            logger.error(f"❌ Veritabanı motoru oluşturulamadı: {e}")
            self.engine = None
        except Exception as e:
            logger.error(f"❌ Beklenmedik bir hata oluştu: {e}")
            self.engine = None
            
    def test_connection(self):
        """Veritabanı bağlantısını test eder."""
        if not self.engine:
            logger.error("Bağlantı motoru mevcut değil, test başarısız.")
            return False
            
        try:
            with self.engine.connect() as connection:
                logger.info("✅ Veritabanı bağlantısı başarıyla test edildi!")
                return True
        except SQLAlchemyError as e:
            logger.error(f"❌ Veritabanı bağlantı testi başarısız: {e}")
            logger.warning("📝 Lütfen veritabanı sunucunuzun çalıştığından ve config ayarlarınızın doğru olduğundan emin olun.")
            return False

    def save_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace'):
        """
        Pandas DataFrame'ini veritabanındaki bir tabloya kaydeder.
        
        Args:
            df (pd.DataFrame): Kaydedilecek DataFrame.
            table_name (str): Veritabanındaki tablo adı.
            if_exists (str): Tablo zaten varsa ne yapılacağı ('fail', 'replace', 'append').
        """
        if self.engine is None:
            logger.error("Veritabanı motoru yok, veri kaydedilemedi.")
            return
            
        if df.empty:
            logger.warning(f"'{table_name}' tablosuna kaydedilecek veri bulunmuyor (DataFrame boş).")
            return

        logger.info(f"💾 DataFrame '{table_name}' tablosuna kaydediliyor ({len(df)} satır)...")
        
        try:
            # DataFrame'i SQL'e yaz
            df.to_sql(
                name=table_name,
                con=self.engine,
                if_exists=if_exists,
                index=False,
                chunksize=1000  # Büyük DataFrame'ler için performansı artırır
            )
            logger.info(f"✅ Veri '{table_name}' tablosuna başarıyla kaydedildi.")
            
        except SQLAlchemyError as e:
            logger.error(f"❌ Veri kaydetme hatası: {e}")
        except Exception as e:
            logger.error(f"❌ Beklenmedik bir hata oluştu: {e}")
            
    def read_sql_to_dataframe(self, sql_query: str) -> Optional[pd.DataFrame]:
        """
        Bir SQL sorgusunu çalıştırır ve sonucu DataFrame olarak döndürür.
        
        Args:
            sql_query (str): Çalıştırılacak SQL sorgusu.
            
        Returns:
            Optional[pd.DataFrame]: Sorgu sonucu veya hata durumunda None.
        """
        if self.engine is None:
            logger.error("Veritabanı motoru yok, sorgu çalıştırılamadı.")
            return None
            
        logger.info(f"🔍 SQL sorgusu çalıştırılıyor: {sql_query[:100]}...")
        
        try:
            with self.engine.connect() as connection:
                df = pd.read_sql(text(sql_query), connection)
                logger.info(f"✅ Sorgu başarıyla çalıştırıldı, {len(df)} satır döndü.")
                return df
        except SQLAlchemyError as e:
            logger.error(f"❌ SQL sorgu hatası: {e}")
            return None

    def execute_sql(self, sql_query: str):
        """
        Sonuç döndürmeyen bir SQL sorgusu çalıştırır (CREATE, INSERT, UPDATE vb.).
        
        Args:
            sql_query (str): Çalıştırılacak SQL sorgusu.
        """
        if self.engine is None:
            logger.error("Veritabanı motoru yok, sorgu çalıştırılamadı.")
            return
            
        logger.info(f"⚙️ SQL sorgusu çalıştırılıyor: {sql_query[:100]}...")
        
        try:
            with self.engine.connect() as connection:
                with connection.begin(): # Transaction başlat
                    connection.execute(text(sql_query))
                logger.info("✅ SQL sorgusu başarıyla çalıştırıldı.")
        except SQLAlchemyError as e:
            logger.error(f"❌ SQL sorgu hatası: {e}")


def test_db_connection():
    """
    Basit bir veritabanı bağlantı testi yapar ve sonucu yazdırır.
    Bu fonksiyon, main.py'deki menüden çağrılmak için tasarlanmıştır.
    """
    logger.info("Veritabanı bağlantı testi başlatılıyor...")
    try:
        db_manager = DatabaseManager()
        # test_connection metodu __init__ içinde zaten çağrılıyor ve sonucu logluyor.
        # Eğer motor başarıyla oluşturulursa, bağlantı başarılıdır.
        if db_manager.engine:
            print("\n[SONUÇ] ✅ Veritabanı bağlantısı başarılı!")
        else:
            print("\n[SONUÇ] ❌ Veritabanı bağlantısı BAŞARISIZ! Lütfen logları ve config dosyasını kontrol edin.")
    except Exception as e:
        logger.error(f"Test sırasında beklenmedik bir hata oluştu: {e}")
        print(f"\n[SONUÇ] ❌ Test sırasında bir hata oluştu: {e}")

if __name__ == '__main__':
    # Bu betik doğrudan çalıştırıldığında basit bir test yapar.
    test_db_connection() 