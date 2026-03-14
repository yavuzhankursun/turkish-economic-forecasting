"""
Tam Otonom Tahmin Sistemi
=========================

Bu modül ARIMA modelini News API ve Claude API ile entegre eder.
TÜBİTAK projesi kapsamında güncel haberlerle beslenen tahmin sistemi.

Geliştirici: [Öğrenci Adı] - [Üniversite]
Danışman: [Danışman Adı]
"""

import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional

# Proje kök dizinini import yolu olarak ekle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models.arima_model import ARIMAForecaster, load_complete_data_from_mongodb
from src.services.news_api_service import NewsAPIService, ClaudeAPIService
from src.data_collection.turkish_financial_scraper import run_scraper

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AutonomousForecastingSystem:
    """Tam otonom tahmin sistemi."""
    
    def __init__(self):
        self.news_service = NewsAPIService()
        self.claude_service = ClaudeAPIService()
        self.arima_forecaster = None
        
    def run_full_system(self) -> Dict:
        """
        Tam otonom sistemi çalıştırır:
        1. Veri çekme ve kaydetme
        2. ARIMA model eğitimi
        3. News API'den haber çekme
        4. Claude ile analiz
        5. Çarpanlı tahmin
        """
        logger.info("🚀 TAM OTONOM TAHMİN SİSTEMİ BAŞLATILIYOR...")
        
        results = {
            "data_collection": None,
            "arima_model": None,
            "news_analysis": None,
            "final_forecast": None,
            "success": False
        }
        
        try:
            # 1. VERİ TOPLAMA VE KAYDETME
            logger.info("📊 1. ADIM: Veri toplama ve kaydetme...")
            results["data_collection"] = self._collect_and_save_data()
            
            # 2. ARIMA MODEL EĞİTİMİ
            logger.info("🤖 2. ADIM: ARIMA model eğitimi...")
            results["arima_model"] = self._train_arima_model()
            
            # 3. HABER ANALİZİ
            logger.info("📰 3. ADIM: Haber analizi...")
            results["news_analysis"] = self._analyze_news()
            
            # 4. ÇARPANLI TAHMİN
            logger.info("🎯 4. ADIM: Çarpanlı tahmin...")
            results["final_forecast"] = self._generate_enhanced_forecast(
                results["arima_model"], 
                results["news_analysis"]
            )
            
            results["success"] = True
            logger.info("✅ TAM OTONOM SİSTEM BAŞARIYLA TAMAMLANDI!")
            
        except Exception as e:
            logger.error(f"❌ Sistem hatası: {e}")
            results["error"] = str(e)
            
        return results
    
    def _collect_and_save_data(self) -> Dict:
        """Veri toplama ve kaydetme işlemi (MongoDB-only)."""
        try:
            logger.info("MongoDB-only veri toplama modu...")
            # MongoDB'den mevcut veriyi kontrol et
            from src.utils.mongodb_manager import MongoDBManager
            
            db_manager = MongoDBManager()
            if db_manager.client is not None and db_manager.database is not None:
                collection = db_manager.database['economic_indicators']
                count = collection.count_documents({})
                
                if count > 0:
                    logger.info(f"MongoDB'de {count} kayıt mevcut, veri toplama atlanıyor.")
                    return {
                        "status": "success",
                        "message": f"MongoDB'de {count} kayıt mevcut"
                    }
                else:
                    logger.info("MongoDB'de veri yok, scraper çalıştırılıyor...")
                    run_scraper()
            else:
                logger.error("MongoDB bağlantısı kurulamadı")
                return {
                    "status": "error", 
                    "message": "MongoDB bağlantısı kurulamadı"
                }
            
            return {
                "status": "success",
                "message": "Veriler başarıyla toplandı ve MongoDB'ye kaydedildi"
            }
        except Exception as e:
            logger.error(f"Veri toplama hatası: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _train_arima_model(self) -> Dict:
        """ARIMA model eğitimi."""
        try:
            # MongoDB'den veri yükle
            data = load_complete_data_from_mongodb(target_field='usd_try')
            
            if data is None or data.empty:
                raise Exception("MongoDB'den veri yüklenemedi")
            
            logger.info(f"ARIMA için {len(data)} kayıt yüklendi")
            
            # ARIMA modeli oluştur ve eğit
            forecaster = ARIMAForecaster(target_column='usd_try')
            forecaster.fit(data, test_size=0.2)
            
            # Performansı değerlendir
            forecaster.evaluate()
            
            # Temel tahmin yap
            forecast_results = forecaster.forecast(steps=12)
            
            if forecast_results:
                self.arima_forecaster = forecaster
                
                return {
                    "status": "success",
                    "model_params": forecaster.best_params,
                    "forecast": forecast_results['forecast'].to_dict(),
                    "message": "ARIMA modeli başarıyla eğitildi"
                }
            else:
                raise Exception("ARIMA tahmin yapılamadı")
                
        except Exception as e:
            logger.error(f"ARIMA model hatası: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _analyze_news(self) -> Dict:
        """Haber analizi."""
        try:
            # News API'den haber çek
            articles = self.news_service.fetch_economic_news(days_back=7)
            
            if not articles:
                logger.warning("Haber çekilemedi, nötr çarpan kullanılıyor")
                return {
                    "status": "warning",
                    "multiplier": 1.0,
                    "analysis": "Haber çekilemedi",
                    "message": "Nötr çarpan (1.0) kullanılıyor"
                }
            
            # Haberleri Claude için formatla
            news_text = self.news_service.format_news_for_claude(articles)
            
            # Claude ile analiz et
            analysis = self.claude_service.analyze_news_sentiment(news_text)
            
            return {
                "status": "success",
                "articles_count": len(articles),
                "multiplier": analysis.get("multiplier", 1.0),
                "analysis": analysis.get("analysis", "Analiz yapılamadı"),
                "reasoning": analysis.get("reasoning", "Gerekçe yok"),
                "confidence": analysis.get("confidence", 0.5),
                "message": f"Claude analizi tamamlandı. Çarpan: {analysis.get('multiplier', 1.0)}"
            }
            
        except Exception as e:
            logger.error(f"Haber analizi hatası: {e}")
            return {
                "status": "error",
                "multiplier": 1.0,
                "message": str(e)
            }
    
    def _generate_enhanced_forecast(self, arima_result: Dict, news_result: Dict) -> Dict:
        """Çarpanlı tahmin üretimi."""
        try:
            if not self.arima_forecaster:
                raise Exception("ARIMA modeli bulunamadı")
            
            # ARIMA tahminlerini al
            arima_forecast = arima_result.get("forecast", {})
            if not arima_forecast:
                raise Exception("ARIMA tahminleri bulunamadı")
            
            # News çarpanını al
            news_multiplier = news_result.get("multiplier", 1.0)
            
            # Çarpanlı tahminler hesapla
            enhanced_forecast = {}
            for date_str, value in arima_forecast.items():
                enhanced_value = value * news_multiplier
                enhanced_forecast[date_str] = enhanced_value
            
            logger.info(f"🎯 Çarpanlı tahminler hesaplandı (çarpan: {news_multiplier})")
            
            return {
                "status": "success",
                "arima_forecast": arima_forecast,
                "news_multiplier": news_multiplier,
                "enhanced_forecast": enhanced_forecast,
                "news_analysis": news_result.get("analysis", ""),
                "message": f"Çarpanlı tahminler başarıyla üretildi"
            }
            
        except Exception as e:
            logger.error(f"Çarpanlı tahmin hatası: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def print_results(self, results: Dict):
        """Sonuçları yazdırır."""
        if not results["success"]:
            logger.error("❌ Sistem başarısız!")
            return
        
        print("\n" + "="*80)
        print("🎯 TAM OTONOM TAHMİN SİSTEMİ SONUÇLARI")
        print("="*80)
        
        # ARIMA Model Bilgileri
        arima_result = results["arima_model"]
        if arima_result["status"] == "success":
            print(f"\n🤖 ARIMA Model: {arima_result['model_params']}")
            print("📊 Temel Tahminler:")
            for date, value in arima_result["forecast"].items():
                print(f"   {date}: {value:.2f} TRY")
        
        # News Analizi
        news_result = results["news_analysis"]
        if news_result["status"] == "success":
            print(f"\n📰 Haber Analizi:")
            print(f"   📈 Çarpan: {news_result['multiplier']:.3f}")
            print(f"   📝 Analiz: {news_result['analysis']}")
            print(f"   🎯 Güven: {news_result['confidence']:.2f}")
        
        # Final Tahminler
        final_result = results["final_forecast"]
        if final_result["status"] == "success":
            print(f"\n🎯 ÇARPANLI TAHMİNLER (2025-10'den 2026-10'a):")
            print(f"   📊 News Çarpanı: {final_result['news_multiplier']:.3f}")
            print("   📈 Tahminler:")
            
            for date, value in final_result["enhanced_forecast"].items():
                original_value = final_result["arima_forecast"][date]
                change = ((value - original_value) / original_value) * 100
                print(f"      {date}: {value:.2f} TRY ({change:+.1f}%)")
            
            # Son tahmin
            last_date = max(final_result["enhanced_forecast"].keys())
            last_value = final_result["enhanced_forecast"][last_date]
            print(f"\n🏆 SON TAHMİN ({last_date}): {last_value:.2f} TRY")
        
        print("\n" + "="*80)

def test_autonomous_system():
    """Tam otonom sistemi test eder."""
    logger.info("Tam otonom tahmin sistemi test ediliyor...")
    
    system = AutonomousForecastingSystem()
    results = system.run_full_system()
    system.print_results(results)
    
    return results

if __name__ == "__main__":
    test_autonomous_system()
