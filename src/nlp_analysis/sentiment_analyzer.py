"""
Türkçe Haber Duygu Analizi Modülü
=================================

Bu modül haber metinlerinin duygu analizini yapar.
TÜBİTAK projesi kapsamında geliştirilmiştir.

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

import re
import pandas as pd
import numpy as np
from textblob import TextBlob
import logging
from typing import Dict, List

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TurkishSentimentAnalyzer:
    """
    Türkçe metinler için duygu analizi sınıfı.
    Özellikle ekonomi haberleri için optimize edilmiştir.
    """
    
    def __init__(self):
        """Türkçe duygu analizi sınıfını başlatır."""
        
        # Türkçe stop words (gereksiz kelimeler)
        self.stop_words = {
            'bir', 'bu', 'da', 'de', 've', 'ki', 'ile', 'için', 
            'olan', 'olarak', 'var', 'yok', 'gibi', 'kadar',
            'çok', 'daha', 'en', 'hem', 'her', 'hiç', 'ne'
        }
        
        # Ekonomi anahtar kelimeleri
        self.economic_keywords = {
            'para_politikasi': ['faiz', 'repo', 'politika', 'merkez', 'bankası'],
            'enflasyon': ['enflasyon', 'fiyat', 'artış', 'tüfe', 'üfe'],
            'doviz': ['dolar', 'euro', 'kur', 'döviz', 'usd', 'eur'],
            'ekonomi': ['ekonomi', 'büyüme', 'gsyh', 'ihracat', 'ithalat']
        }
        
        # Pozitif ve negatif kelimeler
        self.positive_words = {
            'artış', 'yükseliş', 'büyüme', 'gelişme', 'başarı',
            'iyileşme', 'pozitif', 'olumlu', 'güçlü', 'iyi'
        }
        
        self.negative_words = {
            'düşüş', 'azalma', 'gerileme', 'kriz', 'sorun',
            'olumsuz', 'negatif', 'zayıf', 'kötü', 'risk'
        }
        
        logger.info("Türkçe Duygu Analizi modülü başlatıldı")
    
    def preprocess_text(self, text: str) -> str:
        """
        Türkçe metni ön işleme tabi tutar.
        """
        if not text:
            return ""
        
        # Küçük harfe çevir
        text = text.lower()
        
        # HTML taglarını temizle
        text = re.sub(r'<[^>]+>', '', text)
        
        # URL'leri temizle
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Özel karakterleri temizle (Türkçe karakterleri koru)
        text = re.sub(r'[^\w\sçğıöşüÇĞIİÖŞÜ]', ' ', text)
        
        # Fazla boşlukları tek boşluğa çevir
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords(self, text: str) -> Dict[str, List[str]]:
        """
        Metinden ekonomik anahtar kelimeleri çıkarır.
        """
        text = self.preprocess_text(text)
        found_keywords = {}
        
        for category, keywords in self.economic_keywords.items():
            found_in_category = []
            for keyword in keywords:
                if keyword in text:
                    found_in_category.append(keyword)
            
            if found_in_category:
                found_keywords[category] = found_in_category
        
        return found_keywords
    
    def calculate_sentiment(self, text: str) -> Dict[str, float]:
        """
        Metinin duygu skorunu hesaplar.
        """
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        # Stop words'leri çıkar
        words = [word for word in words if word not in self.stop_words]
        
        positive_count = 0
        negative_count = 0
        total_words = len(words)
        
        if total_words == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        # Kelime bazında duygu tespiti
        for word in words:
            if word in self.positive_words:
                positive_count += 1
            if word in self.negative_words:
                negative_count += 1
        
        # Skorları normalize et
        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        neutral_score = 1.0 - (positive_score + negative_score)
        
        if neutral_score < 0:
            neutral_score = 0
        
        return {
            'positive': round(positive_score, 3),
            'negative': round(negative_score, 3),
            'neutral': round(neutral_score, 3)
        }
    
    def classify_sentiment(self, sentiment_scores: Dict[str, float]) -> str:
        """
        Duygu skorlarına göre sınıflandırma yapar.
        """
        positive = sentiment_scores['positive']
        negative = sentiment_scores['negative']
        
        threshold = 0.1
        
        if positive > negative + threshold:
            return 'positive'
        elif negative > positive + threshold:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_article(self, title: str, content: str, source: str = None) -> Dict:
        """
        Haber makalesinin tam analizini yapar.
        """
        full_text = f"{title} {content}"
        
        # Ekonomik anahtar kelimeleri çıkar
        keywords = self.extract_keywords(full_text)
        
        # Duygu analizi yap
        sentiment_scores = self.calculate_sentiment(full_text)
        sentiment_class = self.classify_sentiment(sentiment_scores)
        
        # Ekonomi ile ilgili mi kontrol et
        is_economic = len(keywords) > 0
        
        result = {
            'source': source,
            'title': title,
            'is_economic': is_economic,
            'keywords': keywords,
            'sentiment_scores': sentiment_scores,
            'sentiment_class': sentiment_class,
            'word_count': len(full_text.split())
        }
        
        return result
    
    def analyze_multiple(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Birden fazla makaleyi analiz eder.
        """
        logger.info(f"{len(articles)} makale analiz ediliyor...")
        
        results = []
        
        for article in articles:
            try:
                result = self.analyze_article(
                    title=article.get('title', ''),
                    content=article.get('content', ''),
                    source=article.get('source', 'unknown')
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Makale analiz edilemedi: {e}")
                continue
        
        df = pd.DataFrame(results)
        logger.info(f"Analiz tamamlandı! {len(df)} makale işlendi")
        
        return df


# Test fonksiyonu
def test_sentiment_analyzer():
    """Duygu analizi modülünü test eder."""
    print("Türkçe Duygu Analizi test ediliyor...")
    
    # Test haberleri
    test_articles = [
        {
            'title': 'Merkez Bankası faiz oranlarını artırdı',
            'content': 'TCMB politika faizini yükseltti. Enflasyonla mücadele.',
            'source': 'AA'
        },
        {
            'title': 'Dolar kuru yeni rekor kırdı',
            'content': 'USD/TRY paritesi yükseldi. Piyasalarda endişe.',
            'source': 'Hürriyet'
        }
    ]
    
    # Analyzer'ı başlat
    analyzer = TurkishSentimentAnalyzer()
    
    # Analiz yap
    results = analyzer.analyze_multiple(test_articles)
    
    print("Analiz Sonuçları:")
    for _, row in results.iterrows():
        print(f"Başlık: {row['title']}")
        print(f"Duygu: {row['sentiment_class']}")
        print(f"Ekonomik: {row['is_economic']}")
        print("---")
    
    print("Test tamamlandı!")
    return analyzer

if __name__ == "__main__":
    test_sentiment_analyzer() 