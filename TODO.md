# TÜBİTAK Ekonomik Göstergeler Tahmin Projesi - TODO Listesi

## 📋 Proje Genel Bilgileri
- **Proje Adı:** Türkiye Ekonomik Göstergeler ARIMA Tahmin Sistemi
- **Süre:** 8 Ay
- **Metodoloji:** Agile (2 haftalık sprintler)
- **Yaklaşım:** Test Güdümlü Geliştirme (TDD)

## 🎯 Ana Hedefler
- [~] Döviz kuru, enflasyon ve faiz oranlarının ARIMA modeli ile tahmini (**Temel model tamamlandı**)
- [ ] Siyasi gündem olaylarının ekonomik etki analizi
- [x] Gerçek zamanlı veri toplama ve işleme sistemi (**Manuel tetikleme ile**)
- [ ] İnteraktif görselleştirme ve raporlama

---

## 📊 1. Veri Toplama ve Düzenleme (1-2 Ay) - ✔️ TAMAMLANDI

### 1.1 Veri Kaynakları Kurulumu
- [x] **TCMB Veri Entegrasyonu (Web Scraping ile)**
  - [x] Döviz Kuru (USD/TRY)
  - [x] Repo Faizi
- [x] **TÜİK Veri Entegrasyonu (Web Scraping ile)**
  - [x] Enflasyon (TÜFE - Tüketici Fiyatları)
- [ ] **BDDK Bankacılık Verileri** - *(Sonraki aşama)*

### 1.2 Veri İşleme Altyapısı
- [x] Otomatik veri çekme sistemi (manuel tetikleme ile)
- [x] Verileri PostgreSQL veritabanında depolama
- [x] Veri temizleme algoritmaları
  - [x] Eksik veri interpolasyonu (Basit 'forward fill' metodu ile)
  - [ ] Aykırı değer tespiti ve düzeltme - *(İleri seviye)*
- [ ] Veri tutarlılık kontrolleri
- [ ] Veri günceleme scheduler'ı (cronjob vb.) - *(Otomasyon aşaması)*

### 1.3 Veritabanı Tasarımı
- [x] Veritabanı Teknolojisi Seçimi (PostgreSQL)
- [x] PostgreSQL şema tasarımı (Tek tablo, ham veriler için)
- [x] Veri normalleştirme (Temel seviyede)
- [x] İndeksleme stratejisi (Primary key ile otomatik)
- [ ] Backup ve recovery planı - *(Deployment aşaması)*

---

## 🤖 2. Ekonomik Veri Analizi ve Model Geliştirme (2-3 Ay) - ⏳ DEVAM EDİYOR

### 2.1 ARIMA Model Geliştirme
- [x] **Başlangıç:** ARIMA modeli için temel iskelet oluşturuldu.
- [x] Zaman serisi analizi temel fonksiyonları (Veri yükleme, hazırlama)
- [x] Durağanlık testleri (ADF, KPSS) - *(Fonksiyonu eklendi, aktif kullanılacak)*
- [x] ARIMA parametre optimizasyonu (Basit Grid Search ile)
  - [x] AIC kriterleri hesaplama
  - [ ] BIC kriterleri hesaplama
- [ ] ACF/PACF analizi ile parametre seçimi - *(İyileştirme)*
- [ ] Çoklu değişkenli ARIMA (VARIMA) modeli - *(İleri seviye)*

### 2.2 Model Performans Değerlendirme
- [x] Performans metrikleri hesaplama
  - [x] RMSE (Root Mean Square Error)
  - [x] MAE (Mean Absolute Error)
  - [ ] MAPE (Mean Absolute Percentage Error) - *(Eklenecek)*
- [ ] Çapraz doğrulama (Cross-validation) sistemi - *(İleri seviye)*
- [ ] Model karşılaştırma framework'ü
- [ ] Otomatik model seçim algoritması

### 2.3 Tahmin Sistemi
- [x] Gerçek zamanlı tahmin pipeline'ı (Manuel tetikleme ile)
- [x] Güven aralıkları hesaplama ve görselleştirme
- [ ] Senaryo analizi (pessimistic, optimistic, realistic) - *(İleri seviye)*
- [ ] Model güncelleme otomasyonu - *(Deployment aşaması)*

---

## 🗞️ 3. Siyasi Gündem Takibi ve Analiz Entegrasyonu (3-4 Ay) - ⏳ SIRADAKİ

### 3.1 Haber ve Gündem Takibi
- [ ] Haber sitelerinden otomatik veri çekme
  - [ ] AA, İHA, Sözcü, Hürriyet vb.
  - [ ] Merkez Bankası duyuruları
  - [ ] Bakanlık açıklamaları
- [ ] Web scraping altyapısı
- [ ] RSS feed takip sistemi
- [ ] Social media monitoring (Twitter API)

### 3.2 Doğal Dil İşleme (NLP)
- [ ] Türkçe metin işleme pipeline'ı
- [ ] Duygu analizi (sentiment analysis)
- [ ] Anahtar kelime çıkarma
- [ ] Konu modelleme (topic modeling)
- [ ] Named Entity Recognition (NER)
- [ ] Ekonomik olay sınıflandırması

### 3.3 Etki Analizi Sistemi
- [ ] Siyasi olayların ekonomik etki skorlaması
- [ ] Olay-etki korelasyon analizi
- [ ] Zaman gecikmeli etki analizi
- [ ] Etki büyüklüğü tahmini
- [ ] Model entegrasyon algoritması

---

## 📈 4. Görselleştirme ve Raporlama (4-5 Ay)

### 4.1 İnteraktif Dashboard
- [ ] Streamlit/Dash web uygulaması
- [ ] Gerçek zamanlı veri görselleştirme
- [ ] Interaktif grafik bileşenleri
  - [ ] Zaman serisi grafikleri
  - [ ] Korelasyon ısı haritaları
  - [ ] Tahmin güven aralıkları
  - [ ] Siyasi olay timeline'ı

### 4.2 Grafik Türleri
- [ ] Çizgi grafikleri (zaman serisi)
- [ ] Sütun grafikleri (karşılaştırmalı)
- [ ] Isı haritaları (korelasyon)
- [ ] Box plot'lar (dağılım analizi)
- [ ] Scatter plot'lar (ilişki analizi)

### 4.3 Raporlama Sistemi
- [ ] Otomatik rapor oluşturma
- [ ] PNG/SVG/PDF export fonksiyonları
- [ ] Türkçe ve İngilizce dil desteği
- [ ] Rapor template'leri
- [ ] E-posta otomatik gönderim

---

## 🔧 5. Teknik Altyapı ve Optimizasyon (5-6 Ay)

### 5.1 Performans Optimizasyonu
- [ ] Kod profiling ve bottleneck analizi
- [ ] Memory optimization (64MB altında)
- [ ] Query optimization (2 saniye altında)
- [ ] Caching stratejileri
- [ ] Parallel processing entegrasyonu

### 5.2 Test Altyapısı
- [ ] Unit testler (%90+ coverage)
- [ ] Integration testler
- [ ] Performance testler
- [ ] Load testler
- [ ] CI/CD pipeline kurulumu

### 5.3 Hata Yönetimi ve Loglama
- [ ] Comprehensive logging sistemi
- [ ] Exception handling mekanizmaları
- [ ] Error monitoring ve alerting
- [ ] Recovery stratejileri
- [ ] Health check endpoints

---

## 🧪 6. Test ve Doğrulama (6-7 Ay)

### 6.1 Model Doğrulama
- [ ] Backtest stratejileri
- [ ] Out-of-sample testing
- [ ] Model robustness testleri
- [ ] Stress testing scenarios
- [ ] Benchmark karşılaştırmaları

### 6.2 Sistem Testleri
- [ ] End-to-end testler
- [ ] User acceptance testing
- [ ] Security testing
- [ ] Data integrity testleri
- [ ] Disaster recovery testleri

---

## 📚 7. Dokümantasyon ve Sunum (7-8 Ay)

### 7.1 Teknik Dokümantasyon
- [ ] API dokümantasyonu
- [ ] Code documentation
- [ ] Deployment guide
- [ ] User manual
- [ ] Troubleshooting guide

### 7.2 Akademik Çıktılar
- [ ] Proje raporu hazırlama
- [ ] Bilimsel makale taslağı
- [ ] Konferans sunumu hazırlığı
- [ ] Demo video çekimi
- [ ] GitHub repository hazırlama

---

## 🛠️ Teknoloji Stack'i

### Backend
- [ ] Python 3.9+
- [ ] pandas, numpy (veri işleme)
- [ ] statsmodels (ARIMA)
- [ ] scikit-learn (ML)
- [ ] beautifulsoup4, requests (web scraping)
- [ ] spacy, nltk (NLP)
- [ ] SQLAlchemy (ORM)

### Frontend
- [ ] Streamlit (web dashboard)
- [ ] matplotlib, seaborn (görselleştirme)
- [ ] plotly (interaktif grafikler)

### Database
- [x] PostgreSQL (production)
- [ ] SQLite (development, opsiyonel)

### DevOps
- [ ] pytest (testing)
- [ ] GitHub Actions (CI/CD)
- [ ] Docker (containerization)

---

## 📅 Sprint Planlaması (2 haftalık iterasyonlar)

### Sprint 1-4: Veri Toplama
### Sprint 5-8: Model Geliştirme  
### Sprint 9-12: NLP ve Siyasi Analiz
### Sprint 13-16: Görselleştirme
### Sprint 17-20: Test ve Optimizasyon
### Sprint 21-24: Dokümantasyon

---

## ⚠️ Risk Faktörleri ve B Planları
- [ ] API rate limiting çözümleri
- [ ] Veri kaynağı backup'ları
- [ ] Model performance degradation handling
- [ ] System downtime contingency plans
- [ ] Resource scalability strategies

---

## 📊 Başarı Kriterleri
- [ ] Tahmin doğruluğu: MAPE < %15
- [ ] Sistem response time: < 2 saniye
- [ ] Memory usage: < 64 MB
- [ ] Test coverage: > %90
- [ ] Uptime: > %99.5 