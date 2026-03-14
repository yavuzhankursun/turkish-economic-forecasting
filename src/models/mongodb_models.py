"""
MongoDB Veri Modelleri
======================

Bu modül, MongoDB koleksiyonları için veri modellerini ve şemalarını tanımlar.
Her koleksiyon için standart alanlar ve veri tipleri burada belirlenir.

Proje: TÜBİTAK Ekonomik Göstergeler Tahmin Sistemi
Geliştirici: [Öğrenci Adı]
Tarih: 2024
"""

import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class CurrencyType(Enum):
    """Döviz kuru türleri"""
    USD_TRY = "USD/TRY"
    EUR_TRY = "EUR/TRY"
    GBP_TRY = "GBP/TRY"
    JPY_TRY = "JPY/TRY"

class IndicatorType(Enum):
    """Ekonomik gösterge türleri"""
    EXCHANGE_RATE = "exchange_rate"
    INFLATION = "inflation"
    INTEREST_RATE = "interest_rate"
    UNEMPLOYMENT = "unemployment"
    GDP = "gdp"

class SentimentType(Enum):
    """Duygu analizi türleri"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

@dataclass
class BaseDocument:
    """Tüm belgeler için temel sınıf"""
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Dataclass'ı dictionary'ye dönüştürür"""
        return asdict(self)

@dataclass
class ExchangeRate(BaseDocument):
    """Döviz kuru verisi"""
    date: datetime
    currency_pair: str  # USD/TRY, EUR/TRY vb.
    buy_rate: float
    sell_rate: float
    central_bank_rate: Optional[float] = None
    source: str = "TCMB"
    
    def __post_init__(self):
        if not hasattr(self, 'created_at'):
            self.created_at = datetime.utcnow()
        if not hasattr(self, 'updated_at'):
            self.updated_at = datetime.utcnow()

@dataclass
class InflationData(BaseDocument):
    """Enflasyon verisi"""
    date: datetime
    indicator_type: str  # TÜFE, ÜFE vb.
    value: float
    previous_value: Optional[float] = None
    change_rate: Optional[float] = None
    source: str = "TÜİK"
    
    def __post_init__(self):
        if not hasattr(self, 'created_at'):
            self.created_at = datetime.utcnow()
        if not hasattr(self, 'updated_at'):
            self.updated_at = datetime.utcnow()

@dataclass
class InterestRate(BaseDocument):
    """Faiz oranı verisi"""
    date: datetime
    rate_type: str  # Politika faizi, repo faizi vb.
    value: float
    previous_value: Optional[float] = None
    change_rate: Optional[float] = None
    source: str = "TCMB"
    
    def __post_init__(self):
        if not hasattr(self, 'created_at'):
            self.created_at = datetime.utcnow()
        if not hasattr(self, 'updated_at'):
            self.updated_at = datetime.utcnow()

@dataclass
class NewsArticle(BaseDocument):
    """Haber makalesi"""
    title: str
    content: str
    url: str
    published_at: datetime
    source: str
    author: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    language: str = "tr"
    
    def __post_init__(self):
        if not hasattr(self, 'created_at'):
            self.created_at = datetime.utcnow()
        if not hasattr(self, 'updated_at'):
            self.updated_at = datetime.utcnow()

@dataclass
class SentimentAnalysis(BaseDocument):
    """Duygu analizi sonucu"""
    article_id: str  # NewsArticle'ın _id'si
    text: str
    sentiment_score: float  # -1 ile 1 arası
    sentiment_label: str  # positive, negative, neutral
    confidence: float  # 0 ile 1 arası
    keywords: Optional[List[str]] = None
    model_used: str = "textblob"
    
    def __post_init__(self):
        if not hasattr(self, 'created_at'):
            self.created_at = datetime.utcnow()
        if not hasattr(self, 'updated_at'):
            self.updated_at = datetime.utcnow()

@dataclass
class PoliticalEvent(BaseDocument):
    """Siyasi olay"""
    event_date: datetime
    event_type: str  # Seçim, referandum, hükümet değişikliği vb.
    title: str
    description: str
    impact_level: str  # low, medium, high
    affected_indicators: Optional[List[str]] = None
    source: str = "manual"
    
    def __post_init__(self):
        if not hasattr(self, 'created_at'):
            self.created_at = datetime.utcnow()
        if not hasattr(self, 'updated_at'):
            self.updated_at = datetime.utcnow()

@dataclass
class ModelPrediction(BaseDocument):
    """Model tahmini"""
    model_name: str  # ARIMA, LSTM vb.
    indicator_type: str
    prediction_date: datetime
    predicted_value: float
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    actual_value: Optional[float] = None  # Gerçekleşen değer
    error_rate: Optional[float] = None
    model_version: str = "1.0"
    
    def __post_init__(self):
        if not hasattr(self, 'created_at'):
            self.created_at = datetime.utcnow()
        if not hasattr(self, 'updated_at'):
            self.updated_at = datetime.utcnow()

@dataclass
class ModelPerformance(BaseDocument):
    """Model performans metrikleri"""
    model_name: str
    indicator_type: str
    evaluation_date: datetime
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error
    r2_score: Optional[float] = None
    test_data_size: int = 0
    training_data_size: int = 0
    
    def __post_init__(self):
        if not hasattr(self, 'created_at'):
            self.created_at = datetime.utcnow()
        if not hasattr(self, 'updated_at'):
            self.updated_at = datetime.utcnow()

@dataclass
class EconomicIndicator(BaseDocument):
    """Standart ekonomik gösterge - Tüm veriler için ortak format"""
    date: datetime
    created_at: datetime
    updated_at: datetime
    data_type: str = "economic_indicators"  # Veri türü
    source: str = "TurkishFinancialScraper"  # Veri kaynağı
    
    # Scraping modülünden gelen veriler
    # Döviz kurları
    usd_try: Optional[float] = None  # usd_avg'den gelir
    eur_try: Optional[float] = None
    
    # Faiz oranları  
    policy_rate: Optional[float] = None  # lend_rate'den gelir
    
    # Enflasyon
    inflation_rate: Optional[float] = None  # tufe_rate'den gelir
    
    # Tarih bilgileri (scraping'den gelen)
    year: Optional[int] = None
    month: Optional[int] = None
    
    # Diğer göstergeler (gelecekte eklenebilir)
    gdp_growth: Optional[float] = None
    unemployment_rate: Optional[float] = None
    
    # Ek alanlar (esnek yapı için)
    additional_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not hasattr(self, 'created_at'):
            self.created_at = datetime.utcnow()
        if not hasattr(self, 'updated_at'):
            self.updated_at = datetime.utcnow()

# MongoDB koleksiyon şemaları
COLLECTION_SCHEMAS = {
    "exchange_rates": {
        "required_fields": ["date", "currency_pair", "buy_rate", "sell_rate"],
        "indexes": [
            [("date", 1), ("currency_pair", 1)],  # Compound index
            [("date", -1)],  # Date descending
            [("currency_pair", 1)]  # Currency pair ascending
        ],
        "validation_rules": {
            "buy_rate": {"$gte": 0},
            "sell_rate": {"$gte": 0},
            "currency_pair": {"$in": ["USD/TRY", "EUR/TRY", "GBP/TRY", "JPY/TRY"]}
        }
    },
    
    "inflation_data": {
        "required_fields": ["date", "indicator_type", "value"],
        "indexes": [
            [("date", 1), ("indicator_type", 1)],
            [("date", -1)],
            [("indicator_type", 1)]
        ],
        "validation_rules": {
            "value": {"$gte": 0},
            "indicator_type": {"$in": ["TÜFE", "ÜFE", "Core Inflation"]}
        }
    },
    
    "interest_rates": {
        "required_fields": ["date", "rate_type", "value"],
        "indexes": [
            [("date", 1), ("rate_type", 1)],
            [("date", -1)],
            [("rate_type", 1)]
        ],
        "validation_rules": {
            "value": {"$gte": 0},
            "rate_type": {"$in": ["Politika Faizi", "Repo Faizi", "Overnight Faizi"]}
        }
    },
    
    "news_articles": {
        "required_fields": ["title", "content", "url", "published_at", "source"],
        "indexes": [
            [("published_at", -1)],
            [("source", 1)],
            [("title", "text"), ("content", "text")],  # Text index
            [("published_at", 1), ("source", 1)]
        ],
        "validation_rules": {
            "title": {"$minLength": 10},
            "content": {"$minLength": 50},
            "url": {"$regex": "^https?://"}
        }
    },
    
    "sentiment_analysis": {
        "required_fields": ["article_id", "text", "sentiment_score", "sentiment_label"],
        "indexes": [
            [("article_id", 1)],
            [("sentiment_label", 1)],
            [("sentiment_score", 1)],
            [("created_at", -1)]
        ],
        "validation_rules": {
            "sentiment_score": {"$gte": -1, "$lte": 1},
            "sentiment_label": {"$in": ["positive", "negative", "neutral"]},
            "confidence": {"$gte": 0, "$lte": 1}
        }
    },
    
    "political_events": {
        "required_fields": ["event_date", "event_type", "title", "description", "impact_level"],
        "indexes": [
            [("event_date", -1)],
            [("event_type", 1)],
            [("impact_level", 1)],
            [("event_date", 1), ("impact_level", 1)]
        ],
        "validation_rules": {
            "impact_level": {"$in": ["low", "medium", "high"]},
            "event_type": {"$minLength": 3}
        }
    },
    
    "model_predictions": {
        "required_fields": ["model_name", "indicator_type", "prediction_date", "predicted_value"],
        "indexes": [
            [("prediction_date", -1)],
            [("model_name", 1)],
            [("indicator_type", 1)],
            [("model_name", 1), ("indicator_type", 1), ("prediction_date", -1)]
        ],
        "validation_rules": {
            "predicted_value": {"$gte": 0},
            "model_name": {"$minLength": 2},
            "indicator_type": {"$minLength": 2}
        }
    },
    
    "model_performance": {
        "required_fields": ["model_name", "indicator_type", "evaluation_date", "mae", "mse", "rmse", "mape"],
        "indexes": [
            [("evaluation_date", -1)],
            [("model_name", 1)],
            [("indicator_type", 1)],
            [("model_name", 1), ("indicator_type", 1)]
        ],
        "validation_rules": {
            "mae": {"$gte": 0},
            "mse": {"$gte": 0},
            "rmse": {"$gte": 0},
            "mape": {"$gte": 0}
        }
    },
    
    "economic_indicators": {
        "required_fields": ["date", "data_type", "source"],
        "indexes": [
            [("date", -1)],
            [("year", 1), ("month", 1)],
            [("data_type", 1)],
            [("source", 1)],
            [("date", 1), ("data_type", 1)],
            [("usd_try", 1)],
            [("policy_rate", 1)],
            [("inflation_rate", 1)]
        ],
        "validation_rules": {
            "data_type": {"$in": ["economic_indicators", "exchange_rates", "inflation_data"]},
            "source": {"$minLength": 3},
            "usd_try": {"$gte": 0},
            "policy_rate": {"$gte": 0},
            "inflation_rate": {"$gte": 0},
            "year": {"$gte": 2020, "$lte": 2030},
            "month": {"$gte": 1, "$lte": 12}
        }
    }
}

def get_collection_schema(collection_name: str) -> Optional[Dict[str, Any]]:
    """
    Belirtilen koleksiyon için şemayı döndürür.
    
    Args:
        collection_name (str): Koleksiyon adı
        
    Returns:
        Optional[Dict[str, Any]]: Koleksiyon şeması
    """
    return COLLECTION_SCHEMAS.get(collection_name)

def validate_document(collection_name: str, document: Dict[str, Any]) -> bool:
    """
    Belgeyi koleksiyon şemasına göre doğrular.
    
    Args:
        collection_name (str): Koleksiyon adı
        document (Dict[str, Any]): Doğrulanacak belge
        
    Returns:
        bool: Belge geçerli mi
    """
    schema = get_collection_schema(collection_name)
    if not schema:
        return False
    
    # Gerekli alanları kontrol et
    required_fields = schema.get("required_fields", [])
    for field in required_fields:
        if field not in document:
            print(f"❌ Gerekli alan eksik: {field}")
            return False
    
    # Validation kurallarını kontrol et
    validation_rules = schema.get("validation_rules", {})
    for field, rules in validation_rules.items():
        if field in document:
            value = document[field]
            for rule, constraint in rules.items():
                if rule == "$gte" and value < constraint:
                    print(f"❌ {field} değeri {constraint}'den küçük olamaz")
                    return False
                elif rule == "$lte" and value > constraint:
                    print(f"❌ {field} değeri {constraint}'den büyük olamaz")
                    return False
                elif rule == "$in" and value not in constraint:
                    print(f"❌ {field} değeri geçerli değerlerden biri olmalı: {constraint}")
                    return False
                elif rule == "$minLength" and len(str(value)) < constraint:
                    print(f"❌ {field} minimum {constraint} karakter olmalı")
                    return False
                elif rule == "$regex" and not re.match(constraint, str(value)):
                    print(f"❌ {field} formatı geçersiz")
                    return False
    
    return True

# NOT: Mock data kullanımı yasak olduğu için create_sample_data() fonksiyonu kaldırıldı
# Tüm veriler gerçek API'lerden veya MongoDB'den çekilmelidir

if __name__ == "__main__":
    # Şema testi
    print("\n🔍 Şema testleri:")
    for collection_name in COLLECTION_SCHEMAS.keys():
        schema = get_collection_schema(collection_name)
        print(f"  {collection_name}: {len(schema['required_fields'])} gerekli alan")
