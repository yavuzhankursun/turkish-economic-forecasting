"""
TÜBİTAK Ekonomik Göstergeler Tahmin Projesi - Görselleştirme Modülü
================================================================

Bu modül tam otonom sistemin sonuçlarını profesyonel grafiklerle görselleştirir.

Geliştirici: [Öğrenci Adı] - [Üniversite]
Danışman: [Danışman Adı]
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os

# Proje kök dizinini import yolu olarak ekle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models.arima_model import load_complete_data_from_mongodb
from src.services.news_api_service import NewsAPIService, ClaudeAPIService

# Hızlı görselleştirme fallback'i (veri/API yoksa) - opsiyonel modül
try:
    from quick_visualization import create_quick_visualization  # type: ignore # noqa: F401
except ImportError:
    create_quick_visualization = None  # type: ignore

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastingVisualizer:
    """Tahmin sonuçlarını görselleştiren sınıf."""
    
    def __init__(self):
        self.news_service = NewsAPIService()
        self.claude_service = ClaudeAPIService()
        
    def create_comprehensive_forecast_plot(self):
        """Kapsamlı tahmin grafiği oluşturur."""
        logger.info("🎨 Kapsamlı tahmin grafiği oluşturuluyor...")
        
        # Veri yükle
        data = load_complete_data_from_mongodb(target_field='usd_try')
        if data is None or data.empty:
            logger.error("Veri yüklenemedi! (MongoDB boş/erişim hatası olabilir)")
            if create_quick_visualization is not None:
                logger.info("Hızlı görselleştirmeye (fallback) geçiliyor...")
                return create_quick_visualization()
            return
        
        # ARIMA tahminleri
        from src.models.arima_model import ARIMAForecaster
        forecaster = ARIMAForecaster(target_column='usd_try')
        forecaster.fit(data, test_size=0.2)
        arima_results = forecaster.forecast(steps=12)
        
        # News analizi
        try:
            articles = self.news_service.fetch_economic_news(days_back=7)
            news_text = self.news_service.format_news_for_claude(articles)
            news_analysis = self.claude_service.analyze_news_sentiment(news_text)
        except Exception as e:
            logger.warning(f"News/Claude analizinde sorun: {e}. Nötr çarpan ile devam ediliyor.")
            news_analysis = {"multiplier": 1.0, "analysis": "Haber analizi yapılamadı", "confidence": 0.5}
        
        # Çarpanlı tahminler
        multiplier = news_analysis.get('multiplier', 1.0)
        enhanced_forecast = {}
        
        for date_str, value in arima_results['forecast'].items():
            enhanced_value = value * multiplier
            enhanced_forecast[date_str] = enhanced_value
        
        # Grafik oluştur
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Ana grafik - Tarihsel veri ve tahminler
        self._plot_main_forecast(ax1, data, arima_results['forecast'], enhanced_forecast, multiplier)
        
        # Alt grafik - News analizi
        self._plot_news_analysis(ax2, news_analysis, articles)
        
        plt.tight_layout()
        
        # Grafik kaydet
        filename = f"comprehensive_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Grafik kaydedildi: {filename}")
        
        # Grafik göster
        try:
            plt.show()
        except:
            logger.info("Grafik gösterilemedi, dosya olarak kaydedildi.")
        
        return filename
    
    def _plot_main_forecast(self, ax, historical_data, arima_forecast, enhanced_forecast, multiplier):
        """Ana tahmin grafiğini çizer."""
        
        # Tarihsel veri
        ax.plot(historical_data.index, historical_data.iloc[:, 0], 
                'b-', linewidth=2, label='Tarihsel Veri', alpha=0.8)
        
        # ARIMA tahminleri
        arima_dates = list(arima_forecast.keys())
        arima_values = list(arima_forecast.values())
        ax.plot(arima_dates, arima_values, 
                'r--', linewidth=2, label='ARIMA Temel Tahmin', alpha=0.7)
        
        # Çarpanlı tahminler
        enhanced_dates = list(enhanced_forecast.keys())
        enhanced_values = list(enhanced_forecast.values())
        ax.plot(enhanced_dates, enhanced_values, 
                'g-', linewidth=3, label=f'News Çarpanlı Tahmin (×{multiplier:.3f})', alpha=0.9)
        
        # Son değerleri işaretle
        last_historical = historical_data.iloc[-1, 0]
        last_date = historical_data.index[-1]
        ax.scatter([last_date], [last_historical], color='blue', s=100, zorder=5)
        ax.annotate(f'Son Veri: {last_historical:.2f} TRY', 
                   xy=(last_date, last_historical), xytext=(10, 10),
                   textcoords='offset points', fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7))
        
        # Son tahmin
        final_forecast = enhanced_values[-1]
        final_date = enhanced_dates[-1]
        ax.scatter([final_date], [final_forecast], color='green', s=100, zorder=5)
        ax.annotate(f'Son Tahmin: {final_forecast:.2f} TRY', 
                   xy=(final_date, final_forecast), xytext=(10, -20),
                   textcoords='offset points', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7))
        
        # Grafik ayarları
        ax.set_title('🎯 TÜBİTAK Ekonomik Göstergeler - USD/TRY Tahmin Sistemi', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Tarih', fontsize=12)
        ax.set_ylabel('USD/TRY Kuru', fontsize=12)
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Tarih formatı
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Y ekseni aralığı
        all_values = list(historical_data.iloc[:, 0]) + arima_values + enhanced_values
        y_min, y_max = min(all_values), max(all_values)
        y_range = y_max - y_min
        ax.set_ylim(y_min - y_range*0.05, y_max + y_range*0.05)
    
    def _plot_news_analysis(self, ax, news_analysis, articles):
        """News analizi grafiğini çizer."""
        
        multiplier = news_analysis.get('multiplier', 1.0)
        analysis = news_analysis.get('analysis', 'Analiz yok')
        confidence = news_analysis.get('confidence', 0.5)
        
        # Çarpan bar grafiği
        colors = ['red' if multiplier < 1.0 else 'green' if multiplier > 1.0 else 'gray']
        ax.bar(['News Çarpanı'], [multiplier], color=colors, alpha=0.7)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Nötr (1.0)')
        
        # Çarpan değerini göster
        ax.text(0, multiplier + 0.01, f'{multiplier:.3f}', 
               ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Grafik ayarları
        ax.set_title(f'📰 News Analizi - {len(articles)} Haber Analiz Edildi', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Çarpan Değeri', fontsize=12)
        ax.set_ylim(0.8, 1.2)
        ax.grid(True, alpha=0.3)
        
        # Analiz metnini ekle
        analysis_text = f"Analiz: {analysis[:100]}..." if len(analysis) > 100 else f"Analiz: {analysis}"
        ax.text(0.5, 0.95, analysis_text, transform=ax.transAxes, 
               fontsize=10, ha='center', va='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
        
        # Güven skoru
        ax.text(0.5, 0.05, f"Güven Skoru: {confidence:.2f}", transform=ax.transAxes,
               fontsize=10, ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

def create_final_visualization():
    """Nihai görselleştirmeyi oluşturur."""
    logger.info("🚀 Nihai görselleştirme başlatılıyor...")
    
    visualizer = ForecastingVisualizer()
    filename = visualizer.create_comprehensive_forecast_plot()
    
    logger.info("✅ Nihai görselleştirme tamamlandı!")
    return filename

if __name__ == "__main__":
    create_final_visualization()
