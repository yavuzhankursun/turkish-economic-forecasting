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
import warnings
from datetime import datetime

# TensorFlow uyarılarını bastır
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# --- Proje Kök Dizinini Ayarlama ---
# Bu betik proje kök dizininden çalıştırıldığında, 'src' klasöründeki
# modüllerin sorunsuz bir şekilde import edilmesini sağlar.
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    from src.data_collection.turkish_financial_scraper import run_scraper
    from src.models.arima_model import test_arima_model
    from src.models.advanced_forecaster import test_advanced_forecaster
    from src.models.inflation_forecaster import test_inflation_forecaster
    from src.models.interest_rate_forecaster import test_interest_rate_forecaster
    from src.services.news_api_service import test_news_api
    from src.systems.autonomous_forecaster import test_autonomous_system
    from src.core.economic_analyzer import test_economic_analyzer
    from src.visualization.forecast_visualizer import create_final_visualization
    from src.data_collection.tcmb_data_collector import TCMBDataCollector
    from src.nlp_analysis.sentiment_analyzer import test_sentiment_analyzer
    from src.utils.mongodb_manager import test_mongodb_connection
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
    except ValueError:
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


def run_safely(label, func):
    """Her menü eylemini güvenle çalıştırmak için sargı."""
    try:
        logger.info(f"{label} başlıyor...")
        return func()
    except KeyboardInterrupt:
        print("İşlem kullanıcı tarafından iptal edildi.")
    except Exception as e:
        logger.error(f"{label} sırasında hata: {e}")
        print(f"❌ {label} sırasında hata: {e}")


def system_health_check():
    """Temel sistem sağlık kontrolü (ağ çağrısı yapmaz)."""
    try:
        from config.config import NEWS_API_CONFIG, CLAUDE_API_CONFIG
        news_key_ok = bool(NEWS_API_CONFIG.get("api_key")) and NEWS_API_CONFIG.get("api_key") != "YOUR_NEWS_API_KEY_HERE"
        claude_key = CLAUDE_API_CONFIG.get("api_key", "")
        claude_key_ok = bool(claude_key) and claude_key.startswith("sk-ant-")
    except Exception as e:
        news_key_ok = False
        claude_key_ok = False
        logger.warning(f"Config okunamadı: {e}")

    print("\n--- Sistem Sağlık Kontrolü ---")
    # MongoDB
    try:
        from src.utils.mongodb_manager import test_mongodb_connection as _test_mongo
        run_safely("MongoDB bağlantı testi", _test_mongo)
    except Exception as e:
        print(f"MongoDB test fonksiyonuna erişilemedi: {e}")

    # API anahtarları
    print(f"News API anahtarı: {'OK' if news_key_ok else 'Eksik/Geçersiz'}")
    print(f"Claude API anahtarı: {'OK' if claude_key_ok else 'Eksik/Geçersiz'}")
    # Matplotlib görüntüleme uyarısı (başarısız olursa zaten dosyaya kaydediliyor)
    print("Matplotlib: Grafik gösterimi başarısız olursa PNG kaydı yapılır.")


def run_all_smoke_tests():
    """Ağ-ağır işlemleri atlayarak hızlı duman testleri çalıştırır."""
    print("\n--- Hızlı Smoke Testleri ---")
    # 2: ARIMA testi (MongoDB verisi gerektirir)
    run_safely("ARIMA modeli testi (kısa)", test_arima_model)
    # 5: Nihai görselleştirme (veri yoksa fallback)
    run_safely("Nihai görselleştirme", create_final_visualization)

    # 8: MongoDB bağlantısı
    run_safely("MongoDB bağlantısı", test_mongodb_connection)
    print("Smoke testler tamamlandı.")

def collect_real_data():
    """TCMB/TÜİK'ten gerçek veri toplar ve MongoDB'ye yükler."""
    print("\n╔════════════════════════════════════════════════════════════════════════╗")
    print("║              TCMB/TÜİK GERÇEK VERİ TOPLAMA SİSTEMİ                   ║")
    print("╚════════════════════════════════════════════════════════════════════════╝\n")
    print("📊 Toplanacak Gerçek Veriler:")
    print("   ✓ TÜİK Enflasyon Verileri (TÜFE - 2022-2024)")
    print("   ✓ TCMB Politika Faiz Oranları (2022-2024)")
    print("\n⚠️  Bu işlem MongoDB'ye gerçek veri kaydedecek.\n")
    
    confirm = input("Devam etmek istiyor musunuz? (E/H): ").strip().upper()
    
    if confirm == 'E':
        collector = TCMBDataCollector()
        success = collector.save_to_mongodb()
        
        if success:
            print("\n╔════════════════════════════════════════════════════════════════════════╗")
            print("║                   ✅ BAŞARILI! VERİLER KAYDEDİLDİ                    ║")
            print("╚════════════════════════════════════════════════════════════════════════╝")
            print("\n📈 Kaydedilen Veriler:")
            print("   ✓ 34 ay TÜİK enflasyon verisi")
            print("   ✓ 34 ay TCMB faiz oranı verisi")
            print("\n💡 Artık web arayüzünde GERÇEK verilerle tahmin yapabilirsiniz!")
            print("   → Web sunucusunu başlatın: Menüden seçenek 13")
        else:
            print("\n❌ HATA! Veriler kaydedilemedi.")
    else:
        print("\n❌ İşlem iptal edildi.")


def test_all_indicators():
    """TÜBİTAK gereksinimi: Tüm göstergeleri test eder."""
    print("\n╔════════════════════════════════════════════════════════════════════════╗")
    print("║     TÜBİTAK - ÇOK GÖSTERGELİ EKONOMİK ANALİZ SİSTEMİ TESTİ          ║")
    print("╚════════════════════════════════════════════════════════════════════════╝\n")
    
    # Enflasyon testi
    run_safely("Enflasyon ARIMA modeli", test_inflation_forecaster)
    print("\n")
    
    # Faiz testi
    run_safely("Faiz Oranı ARIMA modeli", test_interest_rate_forecaster)
    print("\n")
    
    # Economic Analyzer (tüm göstergeler)
    run_safely("EconomicAnalyzer (3 gösterge birlikte)", test_economic_analyzer)
    
    print("\n╔════════════════════════════════════════════════════════════════════════╗")
    print("║                     TÜM GÖSTERGE TESTLERİ TAMAMLANDI                  ║")
    print("╚════════════════════════════════════════════════════════════════════════╝")


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
        print("║ 3. Gelişmiş Tahmin Modelleri (Prophet, XGBoost, LSTM)                  ║")
        print("║ 4. Tam Otonom Tahmin Sistemi (ARIMA + News + Claude)                    ║")
        print("║ 5. Nihai Tahmin Görselleştirmesi                                       ║")
        print("║ 6. NLP Duygu Analiz Modülünü Test Et                                   ║")

        print("║ 8. MongoDB Bağlantısını Kontrol Et                                     ║")
        print("║ 9. NewsAPI'den Haber Çek                                               ║")
        print("║ 10. Çıkış                                                              ║")
        print("║ 11. Sistem Sağlık Kontrolü                                             ║")
        print("║ 12. Hızlı Smoke Testleri                                               ║")
        print("║ 13. Web Sunucusunu Başlat (Flask)                                      ║")
        print("║ 14. TÜBİTAK - Tüm Göstergeler Analizi (USD+Enflasyon+Faiz)            ║")
        print("║ 15. TCMB/TÜİK Gerçek Veri Toplama (MongoDB'ye Yükle)                  ║")
        print("╚════════════════════════════════════════════════════════════════════════╝")

        try:
            choice = input("Lütfen bir seçenek girin (1-15): ")
        except KeyboardInterrupt:
            print("\nProgramdan çıkılıyor. Hoşça kalın!")
            break

        if choice == '1':
            logger.info("Seçim 1: Veri toplama ve kaydetme işlemi başlatılıyor...")
            run_safely("Veri toplama ve kaydetme", run_scraper)
        elif choice == '2':
            logger.info("Seçim 2: ARIMA modeli test ediliyor...")
            run_safely("ARIMA modeli testi", test_arima_model)
        elif choice == '3':
            logger.info("Seçim 3: Gelişmiş tahmin modelleri test ediliyor...")
            run_safely("Gelişmiş tahmin modelleri", test_advanced_forecaster)
        elif choice == '4':
            logger.info("Seçim 4: Tam otonom tahmin sistemi çalıştırılıyor...")
            run_safely("Tam otonom tahmin sistemi", test_autonomous_system)
        elif choice == '5':
            logger.info("Seçim 5: Nihai tahmin görselleştirmesi oluşturuluyor...")
            run_safely("Nihai tahmin görselleştirme", create_final_visualization)
        elif choice == '6':
            logger.info("Seçim 6: NLP modülü test ediliyor...")
            run_safely("NLP duygu analizi", test_sentiment_analyzer)

        elif choice == '8':
            logger.info("Seçim 8: MongoDB bağlantısı kontrol ediliyor...")
            run_safely("MongoDB bağlantısı", test_mongodb_connection)
        elif choice == '9':
            logger.info("Seçim 9: NewsAPI'den haber çekiliyor...")
            run_safely("NewsAPI'den haber çekme", test_news_api)
        elif choice == '10':
            print("Programdan çıkılıyor. Hoşça kalın!")
            break
        elif choice == '11':
            run_safely("Sistem sağlık kontrolü", system_health_check)
        elif choice == '12':
            run_safely("Hızlı smoke testleri", run_all_smoke_tests)
        elif choice == '13':
            print("\n🌐 Web sunucusu başlatılıyor...")
            print("Tarayıcınızda şu adresi açın: http://localhost:5000")
            print("Durdurmak için Ctrl+C kullanın\n")
            import subprocess
            try:
                subprocess.run([sys.executable, 'app.py'])
            except KeyboardInterrupt:
                print("\nWeb sunucusu durduruldu. Ana menüye dönülüyor...")
            except Exception as e:
                logger.error(f"Web sunucusu başlatılırken hata: {e}")
                print(f"❌ Web sunucusu başlatılamadı: {e}")
        elif choice == '14':
            test_all_indicators()
        elif choice == '15':
            run_safely("TCMB/TÜİK gerçek veri toplama", collect_real_data)
        else:
            print("Geçersiz seçim. Lütfen 1-15 arasında bir numara girin.")


if __name__ == "__main__":
    main_menu()