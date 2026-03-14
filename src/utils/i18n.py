"""
Çoklu Dil Desteği (i18n)
========================

TÜBİTAK Projesi - Türkçe ve İngilizce dil desteği

G-17 Gereksinimi: Üretilen raporlar hem Türkçe hem de İngilizce dil seçeneklerine sahip olmalıdır.

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

from typing import Dict, Optional
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class I18n:
    """
    Çoklu dil desteği sınıfı.
    
    Desteklenen diller:
    - Türkçe (tr)
    - İngilizce (en)
    """
    
    TRANSLATIONS = {
        'tr': {
            # Genel
            'report_title': 'TÜBİTAK Ekonomik Göstergeler Tahmin Raporu',
            'date': 'Tarih',
            'status': 'Durum',
            'success': 'Başarılı',
            'error': 'Hata',
            'metrics': 'Metrikler',
            'forecast': 'Tahmin',
            'historical': 'Tarihsel Veri',
            'analysis': 'Analiz',
            
            # Göstergeler
            'usd_try': 'USD/TRY Kuru',
            'inflation': 'Enflasyon Oranı',
            'interest_rate': 'Faiz Oranı',
            
            # Metrikler
            'rmse': 'RMSE',
            'mae': 'MAE',
            'mape': 'MAPE',
            'last_value': 'Son Değer',
            'data_points': 'Veri Noktası',
            'model_params': 'Model Parametreleri',
            
            # Performans
            'performance': 'Performans',
            'total_time': 'Toplam Süre',
            'target_50s': 'Hedef (<50s)',
            'successful': 'BAŞARILI',
            'exceeded': 'HEDEF AŞILDI',
            
            # Raporlar
            'report_generated': 'Rapor oluşturuldu',
            'export_completed': 'Export tamamlandı',
            
            # Karşılaştırma
            'comparison': 'Karşılaştırma',
            'correlation': 'Korelasyon',
            'trend_similarity': 'Trend Benzerliği',
            'volatility': 'Volatilite',
            
            # Hatalar
            'no_data': 'Veri bulunamadı',
            'analysis_failed': 'Analiz başarısız oldu',
            'model_not_trained': 'Model henüz eğitilmedi'
        },
        'en': {
            # General
            'report_title': 'TÜBİTAK Economic Indicators Forecast Report',
            'date': 'Date',
            'status': 'Status',
            'success': 'Success',
            'error': 'Error',
            'metrics': 'Metrics',
            'forecast': 'Forecast',
            'historical': 'Historical Data',
            'analysis': 'Analysis',
            
            # Indicators
            'usd_try': 'USD/TRY Exchange Rate',
            'inflation': 'Inflation Rate',
            'interest_rate': 'Interest Rate',
            
            # Metrics
            'rmse': 'RMSE',
            'mae': 'MAE',
            'mape': 'MAPE',
            'last_value': 'Last Value',
            'data_points': 'Data Points',
            'model_params': 'Model Parameters',
            
            # Performance
            'performance': 'Performance',
            'total_time': 'Total Time',
            'target_50s': 'Target (<50s)',
            'successful': 'SUCCESSFUL',
            'exceeded': 'TARGET EXCEEDED',
            
            # Reports
            'report_generated': 'Report generated',
            'export_completed': 'Export completed',
            
            # Comparison
            'comparison': 'Comparison',
            'correlation': 'Correlation',
            'trend_similarity': 'Trend Similarity',
            'volatility': 'Volatility',
            
            # Errors
            'no_data': 'No data available',
            'analysis_failed': 'Analysis failed',
            'model_not_trained': 'Model not trained yet'
        }
    }
    
    def __init__(self, language: str = 'tr'):
        """
        I18n'i başlatır.
        
        Args:
            language: Dil kodu ('tr' veya 'en')
        """
        self.language = language if language in ['tr', 'en'] else 'tr'
        logger.info(f"I18n başlatıldı: {self.language}")
    
    def t(self, key: str, **kwargs) -> str:
        """
        Çeviri yapar.
        
        Args:
            key: Çeviri anahtarı
            **kwargs: Format parametreleri
            
        Returns:
            str: Çevrilmiş metin
        """
        translation = self.TRANSLATIONS.get(self.language, {}).get(key, key)
        
        # Format parametreleri varsa uygula
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError):
                logger.warning(f"Format hatası: {key} - {kwargs}")
        
        return translation
    
    def set_language(self, language: str):
        """
        Dili değiştirir.
        
        Args:
            language: Dil kodu ('tr' veya 'en')
        """
        if language in ['tr', 'en']:
            self.language = language
            logger.info(f"Dil değiştirildi: {language}")
        else:
            logger.warning(f"Geçersiz dil kodu: {language}")
    
    def get_available_languages(self) -> list:
        """
        Mevcut dilleri döndürür.
        
        Returns:
            list: Dil kodları listesi
        """
        return list(self.TRANSLATIONS.keys())


# Global instance
_i18n_instance: Optional['I18n'] = None


def get_i18n(language: str = 'tr') -> I18n:
    """
    I18n instance'ını alır veya oluşturur.
    
    Args:
        language: Dil kodu
        
    Returns:
        I18n: I18n instance'ı
    """
    global _i18n_instance
    if _i18n_instance is None or _i18n_instance.language != language:
        _i18n_instance = I18n(language)
    return _i18n_instance


def t(key: str, language: str = 'tr', **kwargs) -> str:
    """
    Çeviri yapmak için kolay kullanım fonksiyonu.
    
    Args:
        key: Çeviri anahtarı
        language: Dil kodu
        **kwargs: Format parametreleri
        
    Returns:
        str: Çevrilmiş metin
    """
    i18n = get_i18n(language)
    return i18n.t(key, **kwargs)


if __name__ == '__main__':
    # Test
    logger.info("I18n test ediliyor...")
    
    i18n_tr = I18n('tr')
    i18n_en = I18n('en')
    
    logger.info(f"TR: {i18n_tr.t('report_title')}")
    logger.info(f"EN: {i18n_en.t('report_title')}")
    
    logger.info("I18n test tamamlandı.")

