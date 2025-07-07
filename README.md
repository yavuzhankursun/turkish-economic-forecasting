# TÜBİTAK Ekonomik Göstergeler Tahmin Projesi

## 📊 Proje Açıklaması

Bu proje, Türkiye'nin temel ekonomik göstergeleri olan döviz kuru, enflasyon ve faiz oranlarının ARIMA modeli ile tahmin edilmesi ve siyasi gündem olaylarının bu göstergeler üzerindeki etkilerinin analiz edilmesi amacıyla geliştirilmiştir.

### 🎯 Proje Hedefleri

- **Veri Toplama**: TCMB, TÜİK ve BDDK'dan otomatik veri çekimi
- **Tahmin Modeli**: ARIMA modeli ile ekonomik gösterge tahmini
- **Duygu Analizi**: Haber metinlerinin NLP ile analizi
- **Görselleştirme**: İnteraktif dashboard ve raporlama
- **Performans**: 2 saniye altında yanıt, 64MB altında bellek kullanımı

## 📋 Proje Bilgileri

- **Geliştirici**: [Öğrenci Adı] - [Üniversite Adı] [Bölüm]
- **Danışman**: [Danışman Adı]
- **TÜBİTAK Proje No**: [Proje Numarası]
- **Başlangıç Tarihi**: 2024
- **Süre**: 8 Ay
- **Metodoloji**: Agile (2 haftalık sprintler)

## 🏗️ Proje Yapısı

```
├── config/                 # Konfigürasyon dosyaları
│   ├── config.py          # Ana konfigürasyon
│   └── environment_template.txt # Çevre değişkenleri şablonu
├── src/                   # Kaynak kod
│   ├── data_collection/   # Veri toplama modülleri
│   │   └── tcmb_collector.py # TCMB veri çekici
│   ├── models/            # Tahmin modelleri
│   │   └── arima_model.py # ARIMA tahmin modeli
│   ├── nlp_analysis/      # Doğal dil işleme
│   │   └── sentiment_analyzer.py # Duygu analizi
│   ├── visualization/     # Görselleştirme
│   └── utils/             # Yardımcı fonksiyonlar
├── data/                  # Veri dosyaları
│   ├── raw/              # Ham veriler
│   ├── processed/        # İşlenmiş veriler
│   └── external/         # Harici veriler
├── tests/                 # Test dosyaları
├── logs/                  # Log dosyaları
├── main.py               # Ana uygulama
├── requirements.txt      # Python bağımlılıkları
├── TODO.md              # Proje görev listesi
└── README.md            # Bu dosya
```

## 🚀 Kurulum

### 1. Projeyi İndirin
```bash
git clone [proje-repo-url]
cd ekonomik-gostergeler-tahmin
```

### 2. Python Sanal Ortam Oluşturun
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 4. Konfigürasyonu Ayarlayın
```bash
# Çevre değişkenlerini ayarlayın (config/environment_template.txt'yi inceleyin)
# TCMB API anahtarınızı edinin: https://evds2.tcmb.gov.tr/
```

### 5. Projeyi Çalıştırın
```bash
python main.py
```

## 🔧 Kullanım

### Ana Menü Seçenekleri

1. **Veri Toplama Testi**: TCMB veri çekme modülünü test eder
2. **ARIMA Model Testi**: Tahmin modeli test edilir
3. **NLP Analiz Testi**: Türkçe duygu analizi test edilir

### Modül Bazında Kullanım

#### TCMB Veri Toplama
```python
from src.data_collection.tcmb_collector import TCMBCollector

collector = TCMBCollector(api_key="your_api_key")
data = collector.get_exchange_rates("01-01-2024", "31-12-2024")
```

#### ARIMA Tahmin
```python
from src.models.arima_model import ARIMAForecaster

forecaster = ARIMAForecaster()
forecaster.fit(data)
forecast = forecaster.forecast(steps=30)
```

#### Duygu Analizi
```python
from src.nlp_analysis.sentiment_analyzer import TurkishSentimentAnalyzer

analyzer = TurkishSentimentAnalyzer()
result = analyzer.analyze_article(title="Başlık", content="İçerik")
```

## 📊 Veri Kaynakları

### TCMB (Türkiye Cumhuriyet Merkez Bankası)
- **API**: https://evds2.tcmb.gov.tr/
- **Veriler**: USD/TRY, EUR/TRY, politika faizi, repo faizi, para arzı
- **Format**: JSON
- **Güncelleme**: Günlük

### TÜİK (Türkiye İstatistik Kurumu)
- **Web**: https://data.tuik.gov.tr/
- **Veriler**: TÜFE, ÜFE, işsizlik, GSYİH
- **Format**: Web scraping
- **Güncelleme**: Aylık

### BDDK (Bankacılık Düzenleme ve Denetleme Kurumu)
- **Web**: https://www.bddk.org.tr/
- **Veriler**: Kredi faizleri, mevduat faizleri
- **Format**: Web scraping
- **Güncelleme**: Haftalık

## 🤖 Metodoloji

### ARIMA Modeli
- **AR (Autoregressive)**: Geçmiş değerlerin etkisi
- **I (Integrated)**: Durağanlık için fark alma
- **MA (Moving Average)**: Geçmiş hataların etkisi
- **Parametre Optimizasyonu**: AIC ve BIC kriterleri
- **Validasyon**: Çapraz doğrulama ve walk-forward testing

### NLP Analizi
- **Dil**: Türkçe
- **Kütüphaneler**: NLTK, TextBlob, spaCy
- **Özellikler**: 
  - Metin ön işleme
  - Anahtar kelime çıkarma
  - Duygu analizi (pozitif, negatif, nötr)
  - Ekonomik sınıflandırma

## 📈 Performans Hedefleri

- **Yanıt Süresi**: < 2 saniye
- **Bellek Kullanımı**: < 64 MB
- **Test Kapsamı**: > %90
- **Tahmin Doğruluğu**: MAPE < %15
- **Sistem Çalışma Oranı**: > %99.5

## 🧪 Test Etme

### Unit Testler
```bash
pytest tests/unit/ -v
```

### Integration Testler
```bash
pytest tests/integration/ -v
```

### Coverage Raporu
```bash
pytest --cov=src --cov-report=html
```

## 📝 Geliştirme Notları

### Sprint Planlaması (2 haftalık)
- **Sprint 1-4**: Veri Toplama Modülleri
- **Sprint 5-8**: ARIMA Model Geliştirme
- **Sprint 9-12**: NLP ve Haber Analizi
- **Sprint 13-16**: Görselleştirme
- **Sprint 17-20**: Test ve Optimizasyon
- **Sprint 21-24**: Dokümantasyon

### Kod Standartları
- **Linting**: flake8
- **Formatting**: black
- **Docstring**: Google style
- **Type Hints**: Python 3.9+

## 🛠️ Teknoloji Stack

### Backend
- **Python**: 3.9+
- **Veri İşleme**: pandas, numpy
- **İstatistik**: statsmodels, scikit-learn
- **NLP**: nltk, spacy, textblob
- **Web Scraping**: requests, beautifulsoup4

### Veritabanı
- **Development**: SQLite
- **Production**: PostgreSQL

### Görselleştirme
- **Grafikler**: matplotlib, seaborn, plotly
- **Dashboard**: Streamlit

### DevOps
- **Test**: pytest
- **CI/CD**: GitHub Actions
- **Containerization**: Docker (opsiyonel)

## 🚨 Bilinen Sorunlar

1. **TCMB API Rate Limiting**: Dakikada 100 istek sınırı
2. **Türkçe NLP**: Sınırlı kaynak, manuel kelime listeleri kullanılıyor
3. **Web Scraping**: Site değişikliklerine karşı hassas

## 🔄 Güncellemeler

### v1.0.0 (2024-01-01)
- ✅ Proje yapısı oluşturuldu
- ✅ TCMB veri çekme modülü
- ✅ Temel ARIMA modeli
- ✅ Türkçe duygu analizi

### Planlanan Güncellemeler
- 🔄 Dashboard geliştirme
- 🔄 Gerçek zamanlı veri akışı
- 🔄 İleri seviye NLP modelleri
- 🔄 Otomatik rapor oluşturma

## 📞 İletişim

- **Geliştirici**: [Öğrenci Email]
- **Danışman**: [Danışman Email]
- **Üniversite**: [Üniversite Email]

## 📄 Lisans

Bu proje TÜBİTAK projesi kapsamında akademik amaçlarla geliştirilmiştir.
Ticari kullanım için izin gereklidir.

## 🙏 Teşekkürler

- TÜBİTAK - Proje desteği
- TCMB - Veri erişimi
- [Üniversite Adı] - Akademik destek
- Açık kaynak toplulukları - Kütüphaneler

---

**Not**: Bu proje eğitim amaçlıdır ve gerçek finansal yatırım kararları için kullanılmamalıdır. 