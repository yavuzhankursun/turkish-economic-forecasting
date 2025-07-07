"""
ARIMA Zaman Serisi Tahmin Modeli
================================

Bu modül ekonomik göstergelerin ARIMA modeli ile tahminini yapar.
TÜBİTAK projesi kapsamında öğrenci tarafından geliştirilmiştir.

Geliştirici: [Öğrenci Adı] - [Üniversite]
Danışman: [Danışman Adı]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm  # pmdarima kütüphanesini içe aktar
import logging
import sys
import os

# Proje kök dizinini import yolu olarak ekle
# Bu, komut satırından çalıştırıldığında 'src' içindeki modülleri bulmasını sağlar
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.database_manager import DatabaseManager

# Matplotlib stili
plt.style.use('seaborn-v0_8-whitegrid')

# Logging ayarları
# Geliştirilmiş logging formatı
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ARIMAForecaster:
    """
    ARIMA modeli ile ekonomik veri tahmini yapan sınıf.
    
    Bu sınıf şunları yapar:
    - Zaman serisi analizi
    - ARIMA parametre optimizasyonu
    - Model eğitimi ve tahmin
    - Performans değerlendirmesi
    """
    
    def __init__(self, target_column: str = 'usd_avg'):
        """
        ARIMA tahmin sınıfını başlatır.
        
        Args:
            target_column (str): Tahmin edilecek sütunun adı.
        """
        self.target_column = target_column
        self.model = None
        self.fitted_model = None
        self.best_params = None
        self.performance_metrics = None
        self.test_data = None
        self.train_data = None # Eğitim verisini saklamak için eklendi
        self.test_predictions = None # Test tahminlerini saklamak için eklendi
        
        logger.info(f"ARIMA Forecaster '{self.target_column}' hedefi için başlatıldı.")
    
    def check_stationarity(self, data):
        """
        Zaman serisinin durağanlığını kontrol eder.
        
        Args:
            data: Kontrol edilecek zaman serisi
            
        Returns:
            bool: Durağan ise True
        """
        # Augmented Dickey-Fuller testi
        result = adfuller(data.dropna())
        p_value = result[1]
        
        # p-value < 0.05 ise durağan
        is_stationary = p_value < 0.05
        
        # Daha açıklayıcı loglama
        if is_stationary:
            logger.info(f"ADF Testi: Zaman serisi durağan görünüyor (p-value: {p_value:.4f}).")
        else:
            logger.info(f"ADF Testi: Zaman serisi durağan değil (p-value: {p_value:.4f}). Fark alma (differencing) gerekebilir.")
        return is_stationary
    
    def find_best_params(self, data):
        """
        En iyi ARIMA parametrelerini `auto_arima` kullanarak bulur.
        
        Args:
            data: Eğitim verisi
            
        Returns:
            pmdarima.ARIMA: Eğitilmiş en iyi model
        """
        logger.info("En iyi ARIMA parametreleri `auto_arima` ile aranıyor...")
        
        # auto_arima, en iyi p,d,q parametrelerini otomatik bulur.
        # stationarity testini kendi içinde yapar (d parametresini belirler)
        # Mevsimsellik olmadığını varsayıyoruz (seasonal=False)
        self.fitted_model = pm.auto_arima(
            data,
            start_p=1,
            start_q=1,
            test='adf',       # Durağanlık için adf testi kullan
            max_p=5,
            max_q=5,
            m=1,              # Mevsimsellik periyodu yok
            d=None,           # d'yi otomatik bulsun
            seasonal=False,   # Mevsimsel değil
            start_P=0,
            D=0,
            trace=True,       # Aramayı konsola yazdır
            error_action='ignore',  
            suppress_warnings=True, 
            stepwise=True     # Daha hızlı arama için
        )
        
        self.best_params = self.fitted_model.get_params()['order']
        logger.info(f"En iyi parametreler bulundu: ARIMA{self.best_params}")
        
        return self.fitted_model
    
    def fit(self, data, test_size=0.2):
        """
        ARIMA modelini eğitir.
        
        Args:
            data: Eğitim verisi
            test_size: Test seti oranı
        """
        logger.info("ARIMA model eğitimi başlıyor...")
        
        if data.empty:
            logger.error("Eğitilecek veri bulunamadı. İşlem durduruluyor.")
            return

        # Frekansın 'MS' (ay başı) olduğunu varsayalım ve ayarlayalım
        # Bu, `statsmodels` kütüphanesinin frekans uyarısını engeller
        if pd.infer_freq(data.index) is None:
            data = data.asfreq('MS')
            logger.info("Veri frekansı 'MS' (Ay Başı) olarak ayarlandı.")

        # Veriyi eğitim ve test setlerine ayır
        split_idx = int(len(data) * (1 - test_size))
        self.train_data = data[:split_idx] # Eğitim verisini değişkene ata
        self.test_data = data[split_idx:]
        
        # Durağanlığı kontrol et
        self.check_stationarity(self.train_data)

        # En iyi parametreleri bul ve modeli eğit
        self.find_best_params(self.train_data)
        
        logger.info("Model eğitimi tamamlandı!")
    
    def forecast(self, steps=12):
        """
        Gelecek değerleri tahmin eder.
        
        Modeli önce test verileriyle günceller, böylece tahminler
        bilinen en son veriden başlar.
        """
        if self.fitted_model is None:
            logger.error("Model henüz eğitilmedi!")
            return None
        
        logger.info(f"{steps} adım ilerisi için tahmin yapılıyor...")
        
        # Modeli test verisiyle GÜNCELLEMEK, gelecek tahminlerinin
        # gerçek verinin sonundan başlamasını sağlar.
        if self.test_data is not None and not self.test_data.empty:
            # NOT: Bu işlem `fitted_model` nesnesini kalıcı olarak değiştirir.
            # Bu, en güncel verilerle tahmin yapmak için doğru bir yaklaşımdır.
             self.fitted_model.update(self.test_data)
             logger.info(f"Model, {len(self.test_data)} adet test verisi ile güncellendi.")

        # Tahmin yap ve güven aralıklarını al
        forecast_values_np, conf_int = self.fitted_model.predict(n_periods=steps, return_conf_int=True)

        # Gelecek tahminleri için tarih indeksi oluştur
        # Model güncellendiği için son tarih `test_data`nın veya `train_data`nın sonudur.
        last_date = self.test_data.index[-1] if self.test_data is not None and not self.test_data.empty else self.train_data.index[-1]
        forecast_index = pd.date_range(start=last_date, periods=steps + 1, freq='MS')[1:]
        
        # Numpy array'lerini tarih indeksli pandas Series'e dönüştür
        forecast_series = pd.Series(forecast_values_np, index=forecast_index)
        lower_bound_series = pd.Series(conf_int[:, 0], index=forecast_index)
        upper_bound_series = pd.Series(conf_int[:, 1], index=forecast_index)

        results = {
            'forecast': forecast_series,
            'lower_bound': lower_bound_series,
            'upper_bound': upper_bound_series
        }
        
        logger.info("Tahmin başarıyla tamamlandı!")
        return results
    
    def evaluate(self):
        """
        Model performansını test seti üzerinde değerlendirir ve test tahminlerini saklar.
        """
        if self.fitted_model is None or self.test_data is None or self.test_data.empty:
            logger.error("Model eğitilmediği veya test verisi olmadığı için değerlendirme yapılamadı.")
            return None
        
        logger.info(f"Model performansı test seti üzerinde değerlendiriliyor ({len(self.test_data)} adım)...")
        
        # Test seti üzerinde tahmin yap ve sakla
        test_forecast_values = self.fitted_model.predict(n_periods=len(self.test_data))
        self.test_predictions = pd.Series(test_forecast_values, index=self.test_data.index)
        
        # Metrikleri hesapla
        mae = mean_absolute_error(self.test_data, self.test_predictions)
        mse = mean_squared_error(self.test_data, self.test_predictions)
        rmse = np.sqrt(mse)
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }
        
        self.performance_metrics = metrics
        
        logger.info(f"Model Performansı ({self.target_column}):")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def plot_results(self, forecast_results=None):
        """
        Sonuçları görselleştirir.
        """
        if self.fitted_model is None:
            logger.error("Model eğitilmediği için sonuçlar çizdirilemedi.")
            return

        plt.figure(figsize=(15, 8))
        
        # 1. Gerçek veriyi çiz (Eğitim + Test)
        full_data = pd.concat([self.train_data, self.test_data])
        plt.plot(full_data.index, full_data, color='black', alpha=0.7, label='Gerçek Veri')
        
        # 2. Test seti ve gelecek tahminlerini birleştirerek tek bir sürekli çizgi oluştur
        if self.performance_metrics and self.test_predictions is not None:
            # Değerlendirme aşamasında saklanan test tahminlerini kullan
            test_predictions_series = self.test_predictions

            # Gelecek tahminleri
            future_predictions_series = forecast_results['forecast']

            # İki tahmini birleştir
            combined_forecast = pd.concat([test_predictions_series, future_predictions_series])
            
            # Birleştirilmiş tahmini çiz
            plt.plot(combined_forecast.index, combined_forecast, color='blue', linestyle='--', marker='o', markersize=3,
                     label=f'Model Tahmini (Test + Gelecek) - RMSE: {self.performance_metrics["RMSE"]:.2f}')
        
        # 3. Sadece gelecek tahmini için güven aralığını çiz
        if forecast_results and 'forecast' in forecast_results:
            forecast_index = forecast_results['forecast'].index
            
            plt.fill_between(forecast_index,
                             forecast_results['lower_bound'],
                             forecast_results['upper_bound'],
                             alpha=0.2, color='blue', label='Güven Aralığı (%95)')
        
        plt.title(f'{self.target_column.replace("_", " ").upper()} için ARIMA Model Tahminleri', fontsize=18, weight='bold')
        plt.xlabel('Tarih', fontsize=12)
        plt.ylabel(f'Değer ({self.target_column.split("_")[0].upper()})', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()


def load_data_from_db(target_column='usd_avg'):
    """
    Veritabanından belirtilen sütunu yükler ve zaman serisi olarak hazırlar.
    
    Args:
        target_column (str): Yüklenecek hedef sütun adı.
        
    Returns:
        pd.Series: Tarih indeksi ile hazırlanmış zaman serisi verisi.
    """
    logger.info(f"Veritabanından '{target_column}' verisi yükleniyor...")
    db_manager = DatabaseManager()
    
    # Veriyi çek
    query = f"SELECT date, {target_column} FROM financial_data_raw ORDER BY date ASC"
    df = db_manager.read_sql_to_dataframe(query)
    
    if df is None or df.empty:
        logger.error("Veritabanından veri alınamadı veya tablo boş.")
        return None
        
    logger.info(f"{len(df)} satır veri başarıyla çekildi.")
    
    # Veri tiplerini ve indeksi ayarla
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df[target_column].astype(float)
    
    # Eksik verileri doldur (opsiyonel, ileriye/geriye dönük doldurma)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # Frekansı ayarla (aylık başlangıç)
    # Bu, ValueWarning'i önler.
    df = df.asfreq('MS')
    logger.info("Veri frekansı 'MS' (Ay Başı) olarak ayarlandı ve eksik veriler dolduruldu.")

    return df

def test_arima_model():
    """
    ARIMA modelini veritabanı verileriyle test eder.
    """
    logger.info("ARIMA modeli veritabanı verileriyle test ediliyor...")
    
    # Veriyi yükle
    data = load_data_from_db(target_column='usd_avg')
    
    if data is None or data.empty:
        logger.error("Veri yüklenemediği için ARIMA model testi iptal edildi.")
        return

    # Modeli oluştur ve eğit
    forecaster = ARIMAForecaster(target_column='usd_avg')
    forecaster.fit(data, test_size=0.2)
    
    # Performansı değerlendir
    forecaster.evaluate()
    
    # 12 ay sonrası için tahmin yap
    forecast_results = forecaster.forecast(steps=12)
    
    # Sonuçları görselleştir
    if forecast_results:
        forecaster.plot_results(forecast_results)
    
    logger.info("ARIMA model testi başarıyla tamamlandı.")

# Bu dosya doğrudan çalıştırıldığında test fonksiyonunu çağırır
if __name__ == '__main__':
    test_arima_model() 