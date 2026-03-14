"""
News API Servisi
================

Bu modül News API'den ekonomi haberlerini çeker ve Claude API ile analiz eder.
TÜBİTAK projesi kapsamında ARIMA modelini güncel haberlerle beslemek için geliştirilmiştir.

Geliştirici: [Öğrenci Adı] - [Üniversite]
Danışman: [Danışman Adı]
"""

import requests
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
import sys

# Proje kök dizinini import yolu olarak ekle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config.config import NEWS_API_CONFIG, CLAUDE_API_CONFIG

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class NewsAPIService:
    """News API'den ekonomi haberlerini çeken servis."""
    
    def __init__(self):
        self.api_key = NEWS_API_CONFIG["api_key"]
        self.base_url = NEWS_API_CONFIG["base_url"]
        self.language = NEWS_API_CONFIG["language"]
        self.sort_by = NEWS_API_CONFIG["sort_by"]
        self.page_size = NEWS_API_CONFIG["page_size"]
        self.domains = NEWS_API_CONFIG["domains"]
        self.keywords = NEWS_API_CONFIG["keywords"]
        
        if not self.api_key or self.api_key == "YOUR_NEWS_API_KEY_HERE":
            logger.warning("News API anahtarı bulunamadı! Lütfen .env dosyasında NEWS_API_KEY'i ayarlayın.")
    
    def fetch_economic_news(self, days_back: int = 7) -> List[Dict]:
        """
        Son N günün ekonomi haberlerini çeker.
        
        Args:
            days_back: Kaç gün geriye gidileceği
            
        Returns:
            List[Dict]: Haber listesi
        """
        if not self.api_key or self.api_key == "YOUR_NEWS_API_KEY_HERE":
            logger.error("News API anahtarı bulunamadı!")
            return []
        
        logger.info(f"Son {days_back} günün ekonomi haberleri çekiliyor...")
        
        # Tarih aralığı hesapla
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        # API parametreleri
        params = {
            'apiKey': self.api_key,
            'q': ' OR '.join(self.keywords),  # Anahtar kelimeler
            'language': self.language,
            'sortBy': self.sort_by,
            'pageSize': self.page_size,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'domains': self.domains
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                logger.info(f"{len(articles)} ekonomi haberi çekildi.")
                return articles
            else:
                logger.error(f"News API hatası: {data.get('message', 'Bilinmeyen hata')}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"News API bağlantı hatası: {e}")
            return []
        except Exception as e:
            logger.error(f"News API genel hatası: {e}")
            return []
    
    def format_news_for_claude(self, articles: List[Dict]) -> str:
        """
        Haberleri Claude API için formatlar.
        
        Args:
            articles: Haber listesi
            
        Returns:
            str: Formatlanmış haber metni
        """
        if not articles:
            return "Haber bulunamadı."
        
        formatted_news = "EKONOMİ HABERLERİ ANALİZİ:\n\n"
        
        for i, article in enumerate(articles[:10], 1):  # İlk 10 haberi al
            title = article.get('title', 'Başlık yok')
            description = article.get('description', 'Açıklama yok')
            published_at = article.get('publishedAt', 'Tarih yok')
            source = article.get('source', {}).get('name', 'Kaynak yok')
            
            formatted_news += f"{i}. BAŞLIK: {title}\n"
            formatted_news += f"   AÇIKLAMA: {description}\n"
            formatted_news += f"   KAYNAK: {source}\n"
            formatted_news += f"   TARİH: {published_at}\n\n"
        
        return formatted_news

class ClaudeAPIService:
    """Claude API ile haber analizi yapan servis."""
    
    def __init__(self):
        self.api_key = CLAUDE_API_CONFIG["api_key"]
        self.model = CLAUDE_API_CONFIG["model"]
        self.max_tokens = CLAUDE_API_CONFIG["max_tokens"]
        self.temperature = CLAUDE_API_CONFIG["temperature"]
        
        # Fallback artık yok - tek model kullanılıyor
        self.model_fallbacks = []
        
        if not self.api_key or self.api_key == "YOUR_CLAUDE_API_KEY_HERE":
            logger.warning("Claude API anahtarı bulunamadı! Lütfen .env dosyasında CLAUDE_API_KEY'i ayarlayın.")
    
    def analyze_news_sentiment(self, news_text: str) -> Dict:
        """
        Haberleri analiz eder ve ARIMA modeli için çarpan üretir.
        
        Args:
            news_text: Formatlanmış haber metni
            
        Returns:
            Dict: Analiz sonucu ve çarpan
        """
        if not self.api_key or self.api_key == "YOUR_CLAUDE_API_KEY_HERE":
            logger.error("Claude API anahtarı bulunamadı!")
            return {"multiplier": 1.0, "analysis": "API anahtarı bulunamadı", "confidence": 0.5}
        
        # API anahtarının formatını kontrol et
        if not self.api_key.startswith('sk-ant-'):
            logger.warning(f"Claude API anahtarı yanlış format: {self.api_key[:10]}...")
            logger.warning("Demo modunda çalışıyor - nötr çarpan kullanılıyor")
            return {"multiplier": 1.0, "analysis": "Demo modu - API anahtarı yanlış format", "confidence": 0.5}
        
        logger.info("Haberler Claude API ile analiz ediliyor...")
        
        prompt = f"""
Sen bir ekonomist ve finansal analist olarak çalışıyorsun. Aşağıdaki Türkiye ekonomisi haberlerini analiz et ve USD/TRY döviz kuru için bir çarpan üret.

HABERLER:
{news_text}

GÖREV:
1. Haberlerin genel ekonomik durumu nasıl etkilediğini analiz et
2. USD/TRY kuru için pozitif/negatif etkiyi değerlendir
3. 0.8 ile 1.2 arasında bir çarpan üret (0.8 = çok negatif, 1.0 = nötr, 1.2 = çok pozitif)

ÇIKTI FORMATI (JSON):
{{
    "multiplier": 1.05,
    "analysis": "Haberlerin kısa analizi",
    "reasoning": "Çarpanın gerekçesi",
    "confidence": 0.85
}}

Sadece JSON formatında yanıt ver, başka açıklama ekleme.
"""
        
        try:
            headers = {
                'x-api-key': self.api_key,
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            }
            
            data = {
                'model': self.model,
                'max_tokens': self.max_tokens,
                'temperature': self.temperature,
                'messages': [
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            }
            
            logger.info(f"Claude API'ye istek gönderiliyor... Model: {self.model}")
            logger.info(f"API Key başlangıcı: {self.api_key[:10]}...")
            
            # Retry mekanizması (529 hatası için)
            max_retries = 3
            retry_delay = 2  # saniye
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        'https://api.anthropic.com/v1/messages',
                        headers=headers,
                        json=data,
                        timeout=60
                    )
                    
                    logger.info(f"Claude API yanıt kodu: {response.status_code}")
                    
                    # 529 Overloaded hatası
                    if response.status_code == 529:
                        if attempt < max_retries - 1:
                            logger.warning(f"Claude API overloaded (529), {retry_delay}s sonra tekrar deneniyor... (Deneme {attempt + 1}/{max_retries})")
                            import time
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            logger.error("Claude API overloaded, tüm denemeler başarısız. Fallback kullanılacak.")
                            raise Exception("Claude API overloaded (529)")
                    
                    # 404 Model bulunamadı hatası - fallback YOK (tek model zorunlu)
                    if response.status_code == 404:
                        logger.error(f"Claude API 404: Model bulunamadı ({self.model}). Fallback devre dışı, nötr çarpan dönülüyor.")
                        logger.error(f"Hata detayı: {response.text}")
                        return {"multiplier": 1.0, "analysis": "Claude sonnet 4.5 modeli 404 verdi", "confidence": 0.5}
                    
                    # Diğer hatalar
                    if response.status_code != 200:
                        logger.error(f"Claude API hatası: {response.status_code}")
                        logger.error(f"Hata detayı: {response.text}")
                        raise Exception(f"API hatası: {response.status_code}")
                    
                    # Başarılı, döngüden çık
                    break
                    
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        logger.warning(f"Timeout, tekrar deneniyor... (Deneme {attempt + 1}/{max_retries})")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        raise
            
            response.raise_for_status()
            result = response.json()
            
            # Claude'dan gelen yanıtı parse et
            content = result['content'][0]['text']
            
            # Markdown code block'ları temizle (```json ... ``` veya ``` ... ```)
            import re
            # JSON code block'ları bul ve çıkar
            json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            json_match = re.search(json_block_pattern, content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            else:
                # Code block yoksa, sadece JSON objesi bulmaya çalış
                json_obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                json_obj_match = re.search(json_obj_pattern, content, re.DOTALL)
                if json_obj_match:
                    content = json_obj_match.group(0)
            
            # Ön ve son boşlukları temizle
            content = content.strip()
            
            # JSON parse et
            try:
                analysis_result = json.loads(content)
                logger.info(f"Claude analizi tamamlandı. Çarpan: {analysis_result.get('multiplier', 1.0)}")
                return analysis_result
            except json.JSONDecodeError as e:
                logger.error(f"Claude'dan gelen yanıt JSON formatında değil: {e}")
                logger.error(f"İçerik (ilk 500 karakter): {content[:500]}")
                # Son çare: içerikten multiplier ve analysis çıkarmaya çalış
                try:
                    # Basit regex ile multiplier bul
                    multiplier_match = re.search(r'"multiplier"\s*:\s*([0-9.]+)', content)
                    analysis_match = re.search(r'"analysis"\s*:\s*"([^"]+)"', content)
                    
                    multiplier = float(multiplier_match.group(1)) if multiplier_match else 1.0
                    analysis_text = analysis_match.group(1) if analysis_match else "JSON parse hatası - yanıt formatı beklenmedik"
                    
                    logger.warning(f"Regex ile çıkarılan değerler: multiplier={multiplier}, analysis={analysis_text[:50]}")
                    return {
                        "multiplier": multiplier,
                        "analysis": analysis_text,
                        "reasoning": "JSON parse başarısız, regex ile çıkarıldı",
                        "confidence": 0.7
                    }
                except Exception as fallback_error:
                    logger.error(f"Fallback parse da başarısız: {fallback_error}")
                    return {"multiplier": 1.0, "analysis": "JSON parse hatası", "confidence": 0.5}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Claude API bağlantı hatası: {e}")
            return {"multiplier": 1.0, "analysis": f"API hatası: {e}"}
        except Exception as e:
            logger.error(f"Claude API genel hatası: {e}")
            return {"multiplier": 1.0, "analysis": f"Genel hata: {e}"}

def test_news_api():
    """News API servisini test eder."""
    logger.info("News API servisi test ediliyor...")
    
    news_service = NewsAPIService()
    articles = news_service.fetch_economic_news(days_back=7)
    
    if articles:
        logger.info(f"✅ {len(articles)} haber çekildi")
        
        # İlk haberi göster
        if articles:
            first_article = articles[0]
            logger.info(f"İlk haber: {first_article.get('title', 'Başlık yok')}")
        
        # Claude ile analiz et
        claude_service = ClaudeAPIService()
        news_text = news_service.format_news_for_claude(articles)
        analysis = claude_service.analyze_news_sentiment(news_text)
        
        logger.info(f"✅ Claude analizi: Çarpan = {analysis.get('multiplier', 1.0)}")
        logger.info(f"✅ Analiz: {analysis.get('analysis', 'Analiz yok')}")
        
    else:
        logger.error("❌ Haber çekilemedi")

if __name__ == "__main__":
    test_news_api()
