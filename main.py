"""
TÜBİTAK Ekonomik Göstergeler Tahmin Projesi
==========================================

Ana uygulama dosyası. Projenin farklı modüllerini test etmek için
bir komut satırı arayüzü (CLI) sunar.

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

import sys
import os
import logging
from datetime import datetime

# --- Proje Kök Dizinini Ayarlama ---
# Bu betik proje kök dizininden çalıştırıldığında, 'src' klasöründeki
# modüllerin sorunsuz bir şekilde import edilmesini sağlar.
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    from src.data_collection.turkish_financial_scraper import run_scraper
    from src.models.arima_model import test_arima_model
    from src.nlp_analysis.sentiment_analyzer import test_sentiment_analyzer
    from src.utils.database_manager import test_db_connection
except ImportError as e:
    print(f"HATA: Gerekli modüller yüklenemedi. Lütfen proje yapısını kontrol edin.")
    print(f"Detay: {e}")
    sys.exit(1)


# --- Logging Ayarları ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MainApp")


def fetch_newsapi_articles():
    from src.data_collection.newsapi_client import NewsAPIClient
    print("\n--- NewsAPI'den Haber Çekme ---")
    query = input("Aranacak anahtar kelime (örn. ekonomi): ")
    max_articles = input("Kaç haber çekilsin? (varsayılan 10): ")
    try:
        max_articles = int(max_articles)
    except:
        max_articles = 10
    client = NewsAPIClient()
    try:
        articles = client.fetch_news(query, max_articles=max_articles)
        if not articles:
            print("Hiç haber bulunamadı.")
        for i, article in enumerate(articles, 1):
            print(f"\n--- Haber {i} ---")
            print(f"Başlık   : {article['title']}")
            print(f"Açıklama : {article['content']}")
            print(f"Tarih    : {article['publishedAt']}")
            print(f"Kaynak   : {article['source']}")
            print(f"URL      : {article['url']}")
    except Exception as e:
        print(f"NewsAPI'den veri çekilemedi: {e}")


def main_menu():
    """Ana menüyü gösterir ve kullanıcı seçimini yönetir."""
    print("=" * 60)
    print("     TÜBİTAK EKONOMİK GÖSTERGELER TAHMİN SİSTEMİ")
    print("=" * 60)
    print(f"Başlangıç zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    while True:
        print("\n╔════════════════════════════════════════════════════════════════════════╗")
        print("║        TÜBİTAK Ekonomik Göstergeler Tahmin Projesi - Ana Menü        ║")
        print("╠════════════════════════════════════════════════════════════════════════╣")
        print("║ 1. Veri Toplama ve Veritabanına Kaydetme (Scraper)                     ║")
        print("║ 2. ARIMA Modelini Test Et                                              ║")
        print("║ 3. NLP Duygu Analiz Modülünü Test Et                                   ║")
        print("║ 4. Veritabanı Bağlantısını Kontrol Et                                  ║")
        print("║ 5. NewsAPI'den Haber Çek                                               ║")
        print("║ 6. Çıkış                                                               ║")
        print("╚════════════════════════════════════════════════════════════════════════╝")

        choice = input("Lütfen bir seçenek girin (1-6): ")

        if choice == '1':
            logger.info("Seçim 1: Veri toplama ve kaydetme işlemi başlatılıyor...")
            run_scraper()
        elif choice == '2':
            logger.info("Seçim 2: ARIMA modeli test ediliyor...")
            test_arima_model()
        elif choice == '3':
            logger.info("Seçim 3: NLP modülü test ediliyor...")
            test_sentiment_analyzer()
        elif choice == '4':
            logger.info("Seçim 4: Veritabanı bağlantısı kontrol ediliyor...")
            test_db_connection()
        elif choice == '5':
            logger.info("Seçim 5: NewsAPI'den haber çekiliyor...")
            fetch_newsapi_articles()
        elif choice == '6':
            print("Programdan çıkılıyor. Hoşça kalın!")
            break
        else:
            print("Geçersiz seçim. Lütfen 1-6 arasında bir numara girin.")


if __name__ == "__main__":
    main_menu() 