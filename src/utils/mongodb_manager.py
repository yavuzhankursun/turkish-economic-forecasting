"""
MongoDB Veritabanı Yönetim Modülü
=================================

Bu modül, MongoDB veritabanı ile olan tüm etkileşimleri yönetir.
PyMongo kütüphanesi kullanılarak bağlantı kurulur ve veri işlemleri yapılır.

Proje: TÜBİTAK Ekonomik Göstergeler Tahmin Sistemi
Geliştirici: [Öğrenci Adı]
Tarih: 2024
"""

import os
import logging
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List, Any, Union
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, PyMongoError
from pymongo.collection import Collection
from pymongo.database import Database
import sys
from pathlib import Path

# --- Proje Kök Dizinini Path'e Ekleme ---
sys.path.append(str(Path(__file__).resolve().parents[2]))
# -----------------------------------------

# Proje konfigürasyonunu import et
from config.config import MONGODB_CONFIG, MONGODB_COLLECTIONS

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MongoDBManager:
    """
    MongoDB veritabanı işlemlerini yöneten sınıf.
    
    Özellikler:
    - MongoDB bağlantısı kurma ve yönetme
    - Pandas DataFrame'lerini MongoDB'ye kaydetme
    - MongoDB sorguları çalıştırma
    - Bağlantı havuzu (connection pooling)
    - Index yönetimi
    """
    
    def __init__(self, env: str = 'development'):
        """
        MongoDBManager'ı başlatır ve MongoDB bağlantısını oluşturur.
        
        Args:
            env (str): Çalışma ortamı ('development' veya 'production').
                       Bu, config dosyasından hangi ayarların kullanılacağını belirler.
        """
        self.config = MONGODB_CONFIG.get(env)
        if not self.config:
            raise ValueError(f"'{env}' için MongoDB konfigürasyonu bulunamadı!")
            
        self.client: Optional[MongoClient] = None
        self.database: Optional[Database] = None
        self._create_connection()
        
    def _create_connection(self):
        """
        MongoDB bağlantısını oluşturur.
        """
        try:
            # MongoDB connection string oluştur
            if self.config['username'] and self.config['password']:
                # SSL desteği için connection string'i güncelle
                if self.config.get('ssl', False):
                    connection_string = (
                        f"mongodb+srv://{self.config['username']}:{self.config['password']}"
                        f"@{self.config['host']}/{self.config['database']}"
                        f"?retryWrites=true&w=majority"
                    )
                else:
                    connection_string = (
                        f"mongodb://{self.config['username']}:{self.config['password']}"
                        f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
                        f"?authSource={self.config['auth_source']}"
                    )
            else:
                connection_string = f"mongodb://{self.config['host']}:{self.config['port']}"
            
            logger.info(f"🔌 MongoDB bağlantısı oluşturuluyor: {self.config['host']}:{self.config['port']}")
            
            # MongoDB client oluştur
            self.client = MongoClient(
                connection_string,
                maxPoolSize=self.config.get('max_pool_size', 10),
                minPoolSize=self.config.get('min_pool_size', 1),
                maxIdleTimeMS=self.config.get('max_idle_time_ms', 30000),
                serverSelectionTimeoutMS=self.config.get('server_selection_timeout_ms', 5000),
                connectTimeoutMS=self.config.get('connect_timeout_ms', 10000),
                socketTimeoutMS=self.config.get('socket_timeout_ms', 20000),
                retryWrites=self.config.get('retry_writes', True),
                ssl=self.config.get('ssl', False)
            )
            
            # Veritabanını seç
            self.database = self.client[self.config['database']]
            
            # Bağlantıyı test et
            self.test_connection()

        except ConnectionFailure as e:
            logger.error(f"❌ MongoDB bağlantısı kurulamadı: {e}")
            self.client = None
            self.database = None
        except Exception as e:
            logger.error(f"❌ Beklenmedik bir hata oluştu: {e}")
            self.client = None
            self.database = None
            
    def test_connection(self):
        """MongoDB bağlantısını test eder."""
        if not self.client:
            logger.error("MongoDB client mevcut değil, test başarısız.")
            return False
            
        try:
            # Ping komutu ile bağlantıyı test et
            self.client.admin.command('ping')
            logger.info("✅ MongoDB bağlantısı başarıyla test edildi!")
            return True
        except ServerSelectionTimeoutError as e:
            logger.error(f"❌ MongoDB bağlantı testi başarısız: {e}")
            logger.warning("📝 Lütfen MongoDB sunucunuzun çalıştığından ve config ayarlarınızın doğru olduğundan emin olun.")
            return False
        except Exception as e:
            logger.error(f"❌ MongoDB bağlantı testi sırasında hata: {e}")
            return False

    def get_collection(self, collection_name: str) -> Optional[Collection]:
        """
        Belirtilen koleksiyonu döndürür.
        
        Args:
            collection_name (str): Koleksiyon adı
            
        Returns:
            Optional[Collection]: MongoDB koleksiyonu veya None
        """
        if self.database is None:
            logger.error("Veritabanı bağlantısı yok, koleksiyon alınamadı.")
            return None
            
        return self.database[collection_name]

    def insert_document(self, collection_name: str, document: Dict[str, Any]) -> Optional[str]:
        """
        Tek bir belgeyi koleksiyona ekler.
        
        Args:
            collection_name (str): Koleksiyon adı
            document (Dict[str, Any]): Eklenecek belge
            
        Returns:
            Optional[str]: Eklenen belgenin _id'si veya None
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            return None
            
        try:
            # Timestamp ekle
            document['created_at'] = datetime.utcnow()
            document['updated_at'] = datetime.utcnow()
            
            result = collection.insert_one(document)
            logger.info(f"✅ Belge '{collection_name}' koleksiyonuna başarıyla eklendi. ID: {result.inserted_id}")
            return str(result.inserted_id)
        except PyMongoError as e:
            logger.error(f"❌ Belge ekleme hatası: {e}")
            return None

    def insert_many_documents(self, collection_name: str, documents: List[Dict[str, Any]]) -> Optional[List[str]]:
        """
        Birden fazla belgeyi koleksiyona ekler.
        
        Args:
            collection_name (str): Koleksiyon adı
            documents (List[Dict[str, Any]]): Eklenecek belgeler listesi
            
        Returns:
            Optional[List[str]]: Eklenen belgelerin _id'leri veya None
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            return None
            
        try:
            # Timestamp ekle
            current_time = datetime.utcnow()
            for doc in documents:
                doc['created_at'] = current_time
                doc['updated_at'] = current_time
            
            result = collection.insert_many(documents)
            logger.info(f"✅ {len(documents)} belge '{collection_name}' koleksiyonuna başarıyla eklendi.")
            return [str(id) for id in result.inserted_ids]
        except PyMongoError as e:
            logger.error(f"❌ Belge ekleme hatası: {e}")
            return None

    def save_dataframe(self, df: pd.DataFrame, collection_name: str, if_exists: str = 'replace'):
        """
        Pandas DataFrame'ini MongoDB koleksiyonuna kaydeder.
        
        Args:
            df (pd.DataFrame): Kaydedilecek DataFrame.
            collection_name (str): MongoDB koleksiyon adı.
            if_exists (str): Koleksiyon zaten varsa ne yapılacağı ('replace', 'append').
        """
        if self.database is None:
            logger.error("MongoDB bağlantısı yok, veri kaydedilemedi.")
            return
            
        if df.empty:
            logger.warning(f"'{collection_name}' koleksiyonuna kaydedilecek veri bulunmuyor (DataFrame boş).")
            return

        logger.info(f"💾 DataFrame '{collection_name}' koleksiyonuna kaydediliyor ({len(df)} satır)...")
        
        try:
            collection = self.get_collection(collection_name)
            if collection is None:
                return
                
            # DataFrame'i MongoDB belgelerine dönüştür
            documents = df.to_dict('records')
            
            if if_exists == 'replace':
                # Koleksiyonu temizle
                collection.drop()
                logger.info(f"🗑️ '{collection_name}' koleksiyonu temizlendi.")
            
            # Belgeleri ekle
            if documents:
                self.insert_many_documents(collection_name, documents)
                logger.info(f"✅ Veri '{collection_name}' koleksiyonuna başarıyla kaydedildi.")
            
        except Exception as e:
            logger.error(f"❌ Veri kaydetme hatası: {e}")

    def find_documents(self, collection_name: str, query: Dict[str, Any] = None, 
                      limit: int = None, sort: List[tuple] = None) -> List[Dict[str, Any]]:
        """
        Koleksiyondan belgeleri sorgular.
        
        Args:
            collection_name (str): Koleksiyon adı
            query (Dict[str, Any]): MongoDB sorgu filtresi
            limit (int): Maksimum döndürülecek belge sayısı
            sort (List[tuple]): Sıralama kriterleri [(field, direction)]
            
        Returns:
            List[Dict[str, Any]]: Bulunan belgeler
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            return []
            
        try:
            cursor = collection.find(query or {})
            
            if sort:
                cursor = cursor.sort(sort)
                
            if limit:
                cursor = cursor.limit(limit)
                
            documents = list(cursor)
            logger.info(f"🔍 '{collection_name}' koleksiyonunda {len(documents)} belge bulundu.")
            return documents
            
        except PyMongoError as e:
            logger.error(f"❌ Belge sorgulama hatası: {e}")
            return []

    def find_one_document(self, collection_name: str, query: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Koleksiyondan tek bir belgeyi sorgular.
        
        Args:
            collection_name (str): Koleksiyon adı
            query (Dict[str, Any]): MongoDB sorgu filtresi
            
        Returns:
            Optional[Dict[str, Any]]: Bulunan belge veya None
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            return None
            
        try:
            document = collection.find_one(query or {})
            if document:
                logger.info(f"🔍 '{collection_name}' koleksiyonunda belge bulundu.")
            else:
                logger.info(f"🔍 '{collection_name}' koleksiyonunda belge bulunamadı.")
            return document
            
        except PyMongoError as e:
            logger.error(f"❌ Belge sorgulama hatası: {e}")
            return None

    def update_document(self, collection_name: str, query: Dict[str, Any], 
                       update: Dict[str, Any], upsert: bool = False) -> bool:
        """
        Belgeyi günceller.
        
        Args:
            collection_name (str): Koleksiyon adı
            query (Dict[str, Any]): Güncellenecek belgeyi bulmak için sorgu
            update (Dict[str, Any]): Güncelleme verileri
            upsert (bool): Belge bulunamazsa ekle
            
        Returns:
            bool: Güncelleme başarılı mı
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            return False
            
        try:
            # Timestamp ekle
            update['$set'] = update.get('$set', {})
            update['$set']['updated_at'] = datetime.utcnow()
            
            result = collection.update_one(query, update, upsert=upsert)
            
            if result.modified_count > 0:
                logger.info(f"✅ '{collection_name}' koleksiyonunda belge güncellendi.")
                return True
            elif result.upserted_id:
                logger.info(f"✅ '{collection_name}' koleksiyonuna yeni belge eklendi.")
                return True
            else:
                logger.info(f"ℹ️ '{collection_name}' koleksiyonunda güncellenecek belge bulunamadı.")
                return False
                
        except PyMongoError as e:
            logger.error(f"❌ Belge güncelleme hatası: {e}")
            return False

    def delete_documents(self, collection_name: str, query: Dict[str, Any]) -> int:
        """
        Belgeleri siler.
        
        Args:
            collection_name (str): Koleksiyon adı
            query (Dict[str, Any]): Silinecek belgeleri bulmak için sorgu
            
        Returns:
            int: Silinen belge sayısı
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            return 0
            
        try:
            result = collection.delete_many(query)
            logger.info(f"🗑️ '{collection_name}' koleksiyonundan {result.deleted_count} belge silindi.")
            return result.deleted_count
            
        except PyMongoError as e:
            logger.error(f"❌ Belge silme hatası: {e}")
            return 0

    def create_index(self, collection_name: str, index_spec: Union[str, List[tuple]], 
                    unique: bool = False, background: bool = True) -> bool:
        """
        Koleksiyon için index oluşturur.
        
        Args:
            collection_name (str): Koleksiyon adı
            index_spec (Union[str, List[tuple]]): Index spesifikasyonu
            unique (bool): Unique index mi
            background (bool): Background'da oluştur
            
        Returns:
            bool: Index oluşturma başarılı mı
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            return False
            
        try:
            result = collection.create_index(
                index_spec,
                unique=unique,
                background=background
            )
            logger.info(f"✅ '{collection_name}' koleksiyonu için index oluşturuldu: {result}")
            return True
            
        except PyMongoError as e:
            logger.error(f"❌ Index oluşturma hatası: {e}")
            return False

    def get_collection_stats(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Koleksiyon istatistiklerini döndürür.
        
        Args:
            collection_name (str): Koleksiyon adı
            
        Returns:
            Optional[Dict[str, Any]]: Koleksiyon istatistikleri
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            return None
            
        try:
            stats = self.database.command("collStats", collection_name)
            return stats
        except PyMongoError as e:
            logger.error(f"❌ İstatistik alma hatası: {e}")
            return None

    def close_connection(self):
        """MongoDB bağlantısını kapatır."""
        if self.client:
            self.client.close()
            logger.info("🔌 MongoDB bağlantısı kapatıldı.")

    def __enter__(self):
        """Context manager için giriş."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager için çıkış."""
        self.close_connection()


def test_mongodb_connection():
    """
    Basit bir MongoDB bağlantı testi yapar ve sonucu yazdırır.
    Bu fonksiyon, main.py'deki menüden çağrılmak için tasarlanmıştır.
    """
    logger.info("MongoDB bağlantı testi başlatılıyor...")
    try:
        db_manager = MongoDBManager()
        if db_manager.client is not None:
            print("\n[SONUC] [OK] MongoDB baglantisi basarili!")
            
            # Test verisi ekle
            test_collection = "test_collection"
            test_doc = {
                "test_field": "MongoDB test verisi",
                "timestamp": datetime.utcnow(),
                "test_number": 42
            }
            
            # Belge ekle
            doc_id = db_manager.insert_document(test_collection, test_doc)
            if doc_id:
                print(f"[OK] Test belgesi eklendi. ID: {doc_id}")
                
                # Belgeyi sorgula
                found_doc = db_manager.find_one_document(test_collection, {"test_number": 42})
                if found_doc:
                    print("[OK] Test belgesi basariyla sorgulandi.")
                    
                    # Test belgesini sil
                    deleted_count = db_manager.delete_documents(test_collection, {"test_number": 42})
                    if deleted_count > 0:
                        print("[OK] Test belgesi temizlendi.")
                    else:
                        print("[UYARI] Test belgesi silinemedi.")
                else:
                    print("[HATA] Test belgesi sorgulanamadi.")
            else:
                print("[HATA] Test belgesi eklenemedi.")
        else:
            print("\n[SONUC] [HATA] MongoDB baglantisi BASARISIZ! Lutfen loglari ve config dosyasini kontrol edin.")
        
        # Bağlantıyı kapat
        db_manager.close_connection()
                
    except Exception as e:
        logger.error(f"Test sirasinda beklenmedik bir hata olustu: {e}")
        print(f"\n[SONUC] [HATA] Test sirasinda bir hata olustu: {e}")


if __name__ == '__main__':
    # Bu betik doğrudan çalıştırıldığında basit bir test yapar.
    test_mongodb_connection()