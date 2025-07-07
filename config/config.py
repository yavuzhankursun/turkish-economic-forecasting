"""
TÜBİTAK Ekonomik Göstergeler Tahmin Projesi - Konfigürasyon Dosyası
==================================================================

Bu dosya projede kullanılacak tüm konfigürasyon parametrelerini içerir.
API anahtarları, veritabanı ayarları, model parametreleri gibi değerler burada tanımlanır.

Geliştirici: [Öğrenci Adı] - [Üniversite Adı] [Bölüm]
Danışman: [Danışman Adı]
Proje Başlangıç Tarihi: 2024
TÜBİTAK Proje No: [Proje Numarası]
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

# =============================================================================
# TEMEL PROJE AYARLARI
# =============================================================================

# Proje bilgileri - bu bölüm TÜBİTAK raporlarında kullanılacak
PROJECT_INFO = {
    "name": "Türkiye Ekonomik Göstergeler ARIMA Tahmin Sistemi",
    "version": "1.0.0",
    "description": "Döviz kuru, enflasyon ve faiz oranlarının ARIMA modeli ile tahmini ve siyasi etki analizi",
    "author": "[Öğrenci Adı]",
    "supervisor": "[Danışman Adı]",
    "university": "[Üniversite Adı]",
    "department": "[Bölüm Adı]",
    "tubitak_project_no": "[Proje No]",
    "start_date": "2024-01-01",
    "duration_months": 8
}

# Dosya yolları - Windows ve Linux uyumluluğu için pathlib kullanıyoruz
BASE_DIR = Path(__file__).parent.parent  # Proje ana dizini
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src" 
CONFIG_DIR = BASE_DIR / "config"
LOGS_DIR = BASE_DIR / "logs"

# Veri alt dizinleri
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# =============================================================================
# VERİ KAYNAKLARI AYARLARI
# =============================================================================

# TCMB (Türkiye Cumhuriyet Merkez Bankası) API ayarları
TCMB_CONFIG = {
    "base_url": "https://evds2.tcmb.gov.tr/service/evds/",
    "api_key": os.getenv("TCMB_API_KEY", "YOUR_API_KEY_HERE"),  # .env dosyasından okunacak
    "rate_limit": 100,  # dakikada maksimum istek sayısı
    "timeout": 30,  # saniye cinsinden timeout süresi
    "retry_count": 3,  # hata durumunda yeniden deneme sayısı
    
    # Çekilecek veri serileri - TCMB kodları
    "data_series": {
        "usd_try": "TP.DK.USD.A.YTL",  # USD/TRY döviz kuru
        "eur_try": "TP.DK.EUR.A.YTL",  # EUR/TRY döviz kuru
        "policy_rate": "TP.PCTKUR",     # Politika faiz oranı
        "repo_rate": "TP.PCTONUS",      # Repo faiz oranı
        "money_supply": "TP.M3.Y1"      # Para arzı M3
    }
}

# TÜİK (Türkiye İstatistik Kurumu) ayarları
TUIK_CONFIG = {
    "base_url": "https://data.tuik.gov.tr/",
    "scraping_delay": 2,  # istekler arası bekleme süresi (saniye)
    "user_agent": "TubitakEconomicProject/1.0 (Educational Purpose)",
    
    # Çekilecek veri türleri
    "data_types": {
        "inflation_cpi": "tufe",     # Tüketici Fiyat Endeksi
        "inflation_ppi": "ufe",      # Üretici Fiyat Endeksi  
        "unemployment": "issizlik",   # İşsizlik oranları
        "gdp": "gsyih"               # Gayri Safi Yurtiçi Hasıla
    }
}

# BDDK (Bankacılık Düzenleme ve Denetleme Kurumu) ayarları
BDDK_CONFIG = {
    "base_url": "https://www.bddk.org.tr/",
    "data_types": {
        "loan_rates": "kredi_faizleri",
        "deposit_rates": "mevduat_faizleri"
    }
}

# =============================================================================
# ARIMA MODEL AYARLARI
# =============================================================================

# Model parametreleri - bu değerler deneysel olarak optimize edilecek
ARIMA_CONFIG = {
    # Temel ARIMA parametreleri (p, d, q)
    "max_p": 5,  # AR (autoregressive) maksimum değeri
    "max_d": 2,  # Fark alma (differencing) maksimum değeri  
    "max_q": 5,  # MA (moving average) maksimum değeri
    
    # Mevsimsel ARIMA parametreleri (P, D, Q, s)
    "seasonal": {
        "max_P": 2,
        "max_D": 1, 
        "max_Q": 2,
        "s": 12  # mevsimsel periyot (aylık veri için 12)
    },
    
    # Model seçim kriterleri
    "selection_criteria": "aic",  # "aic", "bic", "hqic" seçenekleri
    "information_criterion_threshold": 0.05,
    
    # Tahmin ayarları
    "forecast_horizon": 30,  # kaç gün ilerisini tahmin edeceğiz
    "confidence_levels": [0.80, 0.90, 0.95],  # güven aralıkları
    
    # Model validasyon ayarları
    "test_size": 0.2,  # test seti oranı (%20)
    "cv_folds": 5,     # çapraz doğrulama fold sayısı
    "walk_forward_validation": True  # zaman serisi için önemli
}

# =============================================================================
# NLP (DOĞAL DİL İŞLEME) AYARLARI
# =============================================================================

# Türkçe doğal dil işleme konfigürasyonu
NLP_CONFIG = {
    # Kullanılacak dil modeli
    "language": "tr",  # Türkçe
    "spacy_model": "tr_core_news_md",  # Türkçe spaCy modeli
    
    # Text preprocessing ayarları
    "preprocessing": {
        "remove_punctuation": True,
        "remove_stopwords": True,
        "lowercase": True,
        "lemmatization": True,
        "min_word_length": 3,
        "max_word_length": 20
    },
    
    # Sentiment analysis ayarları
    "sentiment": {
        "model_type": "textblob",  # "textblob", "vader", "custom"
        "neutral_threshold": 0.1,  # nötr duygu eşik değeri
        "positive_threshold": 0.1,
        "negative_threshold": -0.1
    },
    
    # Haber kaynakları ve takip edilecek siteler
    "news_sources": {
        "aa": "https://www.aa.com.tr/tr",
        "iha": "https://www.iha.com.tr/",
        "hurriyet": "https://www.hurriyet.com.tr/",
        "sozcu": "https://www.sozcu.com.tr/",
        "tcmb": "https://www.tcmb.gov.tr/wps/wcm/connect/tr/tcmb+tr/main+menu/duyurular"
    },
    
    # Ekonomik anahtar kelimeler - manuel olarak derlenmiş liste
    "economic_keywords": [
        "faiz", "enflasyon", "döviz", "kur", "ekonomi", "büyüme",
        "işsizlik", "ihracat", "ithalat", "bütçe", "vergi", "yatırım",
        "borsa", "altın", "petrol", "gıda", "enerji", "konut",
        "merkez bankası", "para politikası", "maliye", "tefe", "tüfe"
    ]
}

# =============================================================================
# VERİTABANI AYARLARI
# =============================================================================

# Veritabanı konfigürasyonu - geliştirme için SQLite, production için PostgreSQL
DATABASE_CONFIG = {
    "development": {
        "type": "sqlite",
        "path": str(DATA_DIR / "economic_data.db"),
        "echo": False  # SQL sorguları loglanacak mı?
    },
    
    "production": {
        "type": "postgresql",
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", 5432),
        "database": os.getenv("DB_NAME", "yerdi"),
        "username": os.getenv("DB_USER", "erdinatalihan"),
        "password": os.getenv("DB_PASS", "yeni_sifre"), # TODO: Lütfen burayı kendi şifrenizle güncelleyin
        "pool_size": 10,
        "max_overflow": 20
    }
}

# Tablo isimleri
TABLE_NAMES = {
    "exchange_rates": "exchange_rates",
    "inflation_data": "inflation_data", 
    "interest_rates": "interest_rates",
    "political_events": "political_events",
    "news_articles": "news_articles",
    "model_predictions": "model_predictions",
    "model_performance": "model_performance"
}

# =============================================================================
# GÖRSELLEŞTİRME AYARLARI  
# =============================================================================

# Grafik ve dashboard ayarları
VISUALIZATION_CONFIG = {
    # Matplotlib/Seaborn ayarları
    "figure_size": (12, 8),
    "dpi": 300,  # yüksek çözünürlük için
    "font_size": 12,
    "title_font_size": 16,
    "color_palette": "viridis",  # renk paleti
    
    # Türkçe karakter desteği
    "font_family": "DejaVu Sans",
    "unicode_support": True,
    
    # Export ayarları
    "export_formats": ["png", "svg", "pdf"],
    "export_dpi": 300,
    
    # Dashboard ayarları (Streamlit)
    "dashboard": {
        "title": "Türkiye Ekonomik Göstergeler Dashboard",
        "layout": "wide",
        "theme": "light",
        "auto_refresh": True,
        "refresh_interval": 300  # 5 dakika
    }
}

# =============================================================================
# PERFORMANS VE OPTİMİZASYON AYARLARI
# =============================================================================

# Sistem performans kriterleri - TÜBİTAK gereksinimlerine uygun
PERFORMANCE_CONFIG = {
    "max_response_time": 2.0,  # maksimum yanıt süresi (saniye)
    "max_memory_usage": 64,    # maksimum bellek kullanımı (MB)
    "max_cpu_usage": 80,       # maksimum CPU kullanımı (%)
    
    # Caching ayarları
    "cache_enabled": True,
    "cache_ttl": 3600,  # cache süresi (saniye)
    "cache_size": 100,  # maksimum cache boyutu (MB)
    
    # Parallel processing
    "max_workers": 4,  # paralel işlem sayısı
    "chunk_size": 1000,  # veri parça boyutu
    
    # Logging ve monitoring
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "log_rotation": "1 week",
    "max_log_files": 10
}

# =============================================================================
# GÜVENLİK VE YETKİLENDİRME
# =============================================================================

# Güvenlik ayarları - öğrenci projesi için temel seviye
SECURITY_CONFIG = {
    "api_rate_limiting": True,
    "max_requests_per_minute": 100,
    "session_timeout": 1800,  # 30 dakika
    "password_min_length": 8,
    
    # Data validation
    "input_validation": True,
    "sql_injection_protection": True,
    "xss_protection": True
}

# =============================================================================
# TEST VE KALITE GÜVENCESİ AYARLARI
# =============================================================================

# Test konfigürasyonu - %90+ coverage hedefi
TEST_CONFIG = {
    "target_coverage": 90,  # hedef test kapsamı (%)
    "unit_test_timeout": 30,  # saniye
    "integration_test_timeout": 120,  # saniye
    
    # Test verileri
    "test_data_size": 1000,  # test için kullanılacak veri miktarı
    "mock_api_responses": True,
    
    # Performans testleri
    "load_test_users": 10,
    "stress_test_duration": 300  # saniye
}

# =============================================================================
# GELİŞTİRME ORTAMI AYARLARI
# =============================================================================

# Debug ve development ayarları
DEBUG_CONFIG = {
    "debug_mode": True,  # production'da False olacak
    "verbose_logging": True,
    "profiling_enabled": True,
    "memory_profiling": True,
    
    # Hot reload ayarları
    "auto_reload": True,
    "watch_directories": [str(SRC_DIR), str(CONFIG_DIR)]
}

# Environment variables kontrolü
REQUIRED_ENV_VARS = [
    "TCMB_API_KEY",
    "DB_HOST", 
    "DB_USER",
    "DB_PASS"
]

# =============================================================================
# NEWSAPI AYARLARI
# =============================================================================

NEWSAPI_CONFIG = {
    "api_key": os.getenv("NEWSAPI_KEY", "e3dd5088b7cd4b68982443e2c2dc0db7"),
    "base_url": "https://newsapi.org/v2/",
    "default_language": "tr",
    "page_size": 100
}

def validate_config():
    """
    Konfigürasyon dosyasının doğruluğunu kontrol eder.
    Eksik environment variable'lar varsa uyarı verir.
    
    Bu fonksiyon proje başlangıcında çağrılarak
    gerekli ayarların tamamlandığından emin olunur.
    """
    missing_vars = []
    for var in REQUIRED_ENV_VARS:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Eksik environment variables: {', '.join(missing_vars)}")
        print("📝 Lütfen .env dosyasını kontrol edin veya sistem değişkenlerini ayarlayın.")
        return False
    
    # Dizinlerin varlığını kontrol et
    for directory in [DATA_DIR, LOGS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✅ Konfigürasyon doğrulandı!")
    return True

if __name__ == "__main__":
    # Konfigürasyon test edildiğinde
    print("🔧 Proje konfigürasyonu kontrol ediliyor...")
    if validate_config():
        print("🎉 Tüm ayarlar başarıyla yüklendi!")
    else:
        print("❌ Konfigürasyon hataları mevcut!") 