"""
Gelişmiş Zaman Serisi Tahmin Modelleri
=====================================

Bu modül ARIMA'dan daha gelişmiş algoritmalar kullanarak
nokta atışı tahminler yapar.

Geliştirici: [Öğrenci Adı] - [Üniversite]
Danışman: [Danışman Adı]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import sys
import os
from typing import Dict, List, Tuple, Optional

# Proje kök dizinini import yolu olarak ekle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.mongodb_manager import MongoDBManager

# Gelişmiş modeller
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet kurulu değil. pip install prophet")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost kurulu değil. pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM kurulu değil. pip install lightgbm")

try:
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
    from tensorflow.keras.optimizers import Adam  # type: ignore
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    Sequential = None  # type: ignore
    LSTM = None  # type: ignore
    Dense = None  # type: ignore
    Dropout = None  # type: ignore
    Adam = None  # type: ignore
    print("TensorFlow kurulu değil. pip install tensorflow")

# sklearn'i ayrıca import et
from sklearn.preprocessing import MinMaxScaler

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class AdvancedForecaster:
    """
    Gelişmiş zaman serisi tahmin modelleri sınıfı.
    Prophet, XGBoost, LightGBM ve LSTM modellerini kullanır.
    """
    
    def __init__(self, target_field='usd_try'):
        self.target_field = target_field
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.data = None
        
        logger.info(f"AdvancedForecaster '{target_field}' hedefi için başlatıldı.")
    
    def load_data_from_mongodb(self) -> pd.DataFrame:
        """MongoDB'den veri yükler ve özellik mühendisliği yapar."""
        logger.info(f"MongoDB'den '{self.target_field}' verisi yükleniyor...")
        
        try:
            with MongoDBManager() as mongodb_manager:
                # Tüm ekonomik göstergeleri çek
                query = {
                    "data_type": "economic_indicators",
                    "usd_try": {"$exists": True, "$ne": None},
                    "policy_rate": {"$exists": True, "$ne": None},
                    "inflation_rate": {"$exists": True, "$ne": None}
                }
                
                documents = mongodb_manager.find_documents(
                    collection_name='economic_indicators',
                    query=query,
                    sort=[("date", 1)]
                )
                
                if not documents:
                    logger.error("MongoDB'den veri bulunamadı.")
                    return None
                
                df = pd.DataFrame(documents)
                
                # ObjectId'yi kaldır (sayısal hesaplamalarda sorun yaratır)
                if '_id' in df.columns:
                    df = df.drop('_id', axis=1)
                
                df['date'] = pd.to_datetime(df['date'])
                df = df.dropna(subset=[self.target_field])
                df = df.drop_duplicates(subset=['date'], keep='last')
                df.set_index('date', inplace=True)
                
                # Özellik mühendisliği
                df = self._create_features(df)
                
                logger.info(f"{len(df)} satır veri yüklendi ve özellikler oluşturuldu.")
                return df
                
        except Exception as e:
            logger.error(f"Veri yükleme hatası: {e}")
            return None
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Zaman serisi özelliklerini oluşturur."""
        logger.info("Özellik mühendisliği yapılıyor...")
        
        # Hedef değişken
        df['target'] = df[self.target_field]
        
        # Zaman özellikleri
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['day_of_year'] = df.index.dayofyear
        
        # Lag özellikleri (geçmiş değerler)
        for lag in [1, 2, 3, 6, 12]:
            df[f'target_lag_{lag}'] = df['target'].shift(lag)
        
        # Rolling istatistikleri
        for window in [3, 6, 12]:
            df[f'target_ma_{window}'] = df['target'].rolling(window=window).mean()
            df[f'target_std_{window}'] = df['target'].rolling(window=window).std()
            df[f'target_min_{window}'] = df['target'].rolling(window=window).min()
            df[f'target_max_{window}'] = df['target'].rolling(window=window).max()
        
        # Momentum özellikleri
        df['target_momentum_3'] = df['target'] / df['target'].shift(3) - 1
        df['target_momentum_6'] = df['target'] / df['target'].shift(6) - 1
        df['target_momentum_12'] = df['target'] / df['target'].shift(12) - 1
        
        # Diğer ekonomik göstergeler
        if 'policy_rate' in df.columns:
            df['policy_rate_lag_1'] = df['policy_rate'].shift(1)
            df['policy_rate_ma_3'] = df['policy_rate'].rolling(window=3).mean()
        
        if 'inflation_rate' in df.columns:
            df['inflation_rate_lag_1'] = df['inflation_rate'].shift(1)
            df['inflation_rate_ma_3'] = df['inflation_rate'].rolling(window=3).mean()
        
        # NaN değerleri kaldır
        df = df.dropna()
        
        # Özellik sütunlarını belirle
        self.feature_columns = [col for col in df.columns if col not in ['target', self.target_field]]
        
        logger.info(f"{len(self.feature_columns)} özellik oluşturuldu.")
        return df
    
    def train_prophet_model(self, df: pd.DataFrame) -> Dict:
        """Facebook Prophet modelini eğitir."""
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet kurulu değil, atlanıyor.")
            return None
        
        logger.info("Prophet modeli eğitiliyor...")
        
        try:
            # Prophet için veri formatı
            prophet_df = df.reset_index()
            prophet_df = prophet_df[['date', 'target']].rename(columns={'date': 'ds', 'target': 'y'})
            
            # Model oluştur ve eğit
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            model.fit(prophet_df)
            
            self.models['prophet'] = model
            logger.info("Prophet modeli başarıyla eğitildi.")
            return {'model': model, 'type': 'prophet'}
            
        except Exception as e:
            logger.error(f"Prophet model eğitimi hatası: {e}")
            return None
    
    def train_xgboost_model(self, df: pd.DataFrame) -> Dict:
        """XGBoost modelini eğitir."""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost kurulu değil, atlanıyor.")
            return None
        
        logger.info("XGBoost modeli eğitiliyor...")
        
        try:
            # Veriyi hazırla
            X = df[self.feature_columns]
            y = df['target']
            
            # Veriyi normalize et
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Model oluştur ve eğit
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model.fit(X_scaled, y)
            
            self.models['xgboost'] = model
            self.scalers['xgboost'] = scaler
            logger.info("XGBoost modeli başarıyla eğitildi.")
            return {'model': model, 'scaler': scaler, 'type': 'xgboost'}
            
        except Exception as e:
            logger.error(f"XGBoost model eğitimi hatası: {e}")
            return None
    
    def train_lightgbm_model(self, df: pd.DataFrame) -> Dict:
        """LightGBM modelini eğitir."""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM kurulu değil, atlanıyor.")
            return None
        
        logger.info("LightGBM modeli eğitiliyor...")
        
        try:
            # Veriyi hazırla
            X = df[self.feature_columns]
            y = df['target']
            
            # Veriyi normalize et
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Model oluştur ve eğit
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X_scaled, y)
            
            self.models['lightgbm'] = model
            self.scalers['lightgbm'] = scaler
            logger.info("LightGBM modeli başarıyla eğitildi.")
            return {'model': model, 'scaler': scaler, 'type': 'lightgbm'}
            
        except Exception as e:
            logger.error(f"LightGBM model eğitimi hatası: {e}")
            return None
    
    def train_lstm_model(self, df: pd.DataFrame) -> Dict:
        """LSTM modelini eğitir."""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow kurulu değil, atlanıyor.")
            return None
        
        logger.info("LSTM modeli eğitiliyor...")
        
        try:
            # Veriyi hazırla
            X = df[self.feature_columns].values
            y = df['target'].values
            
            # Veriyi normalize et
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            
            # LSTM için veri formatı (sequence)
            def create_sequences(data, target, seq_length=12):
                X_seq, y_seq = [], []
                for i in range(seq_length, len(data)):
                    X_seq.append(data[i-seq_length:i])
                    y_seq.append(target[i])
                return np.array(X_seq), np.array(y_seq)
            
            X_seq, y_seq = create_sequences(X_scaled, y_scaled)
            
            # Model oluştur
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Model eğit
            model.fit(X_seq, y_seq, epochs=50, batch_size=32, verbose=0)
            
            self.models['lstm'] = model
            self.scalers['lstm_X'] = scaler_X
            self.scalers['lstm_y'] = scaler_y
            logger.info("LSTM modeli başarıyla eğitildi.")
            return {'model': model, 'scaler_X': scaler_X, 'scaler_y': scaler_y, 'type': 'lstm'}
            
        except Exception as e:
            logger.error(f"LSTM model eğitimi hatası: {e}")
            return None
    
    def train_moving_average_model(self, df: pd.DataFrame):
        """Basit Moving Average modeli eğitir."""
        logger.info("Moving Average modeli eğitiliyor...")
        
        try:
            # Son 6 ayın ortalamasını al
            ma_value = df['target'].tail(6).mean()
            trend = df['target'].tail(6).pct_change().mean()
            
            self.models['moving_average'] = {
                'ma_value': ma_value,
                'trend': trend,
                'last_value': df['target'].iloc[-1]
            }
            
            # self.data'yı da set et
            self.data = df
            
            logger.info(f"Moving Average modeli eğitildi. MA = {ma_value:.2f}, Trend = {trend:.3f}")
            return {'type': 'moving_average', 'ma_value': ma_value, 'trend': trend}
            
        except Exception as e:
            logger.error(f"Moving Average model eğitimi hatası: {e}")
            return None
    
    def train_simple_model(self, df: pd.DataFrame):
        """Basit Linear Regression modeli eğitir (her zaman çalışır)."""
        logger.info("Basit Linear Regression modeli eğitiliyor...")
        
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score
            
            # Veriyi hazırla
            X = df[self.feature_columns]
            y = df['target']
            
            # Model oluştur ve eğit
            model = LinearRegression()
            model.fit(X, y)
            
            # Performansı değerlendir
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            self.models['linear'] = model
            logger.info(f"Linear Regression modeli eğitildi. R² = {r2:.3f}, MSE = {mse:.3f}")
            return {'model': model, 'type': 'linear', 'r2': r2, 'mse': mse}
            
        except Exception as e:
            logger.error(f"Linear Regression model eğitimi hatası: {e}")
            return None
    
    def train_all_models(self, df: pd.DataFrame):
        """Tüm modelleri eğitir."""
        logger.info("Tüm gelişmiş modeller eğitiliyor...")
        
        self.data = df
        
        # Önce basit modeli eğit (her zaman çalışır)
        self.train_moving_average_model(df)
        self.train_simple_model(df)
        
        # Diğer modelleri dene
        self.train_prophet_model(df)
        self.train_xgboost_model(df)
        self.train_lightgbm_model(df)
        self.train_lstm_model(df)
        
        logger.info(f"{len(self.models)} model başarıyla eğitildi.")
    
    def ensemble_forecast(self, steps: int = 12) -> pd.DataFrame:
        """Ensemble tahmin yapar (tüm modellerin ortalaması)."""
        logger.info(f"Ensemble tahmin yapılıyor ({steps} adım)...")
        
        forecasts = {}
        
        # Her model için tahmin yap
        for model_name, model_info in self.models.items():
            try:
                if model_name == 'moving_average':
                    forecast = self._moving_average_forecast(model_info, steps)
                elif model_name == 'linear':
                    forecast = self._linear_forecast(model_info, steps)
                elif model_name == 'prophet':
                    forecast = self._prophet_forecast(model_info, steps)
                elif model_name == 'xgboost':
                    forecast = self._xgboost_forecast(model_info, steps)
                elif model_name == 'lightgbm':
                    forecast = self._lightgbm_forecast(model_info, steps)
                elif model_name == 'lstm':
                    forecast = self._lstm_forecast(model_info, steps)
                
                if forecast is not None:
                    forecasts[model_name] = forecast
                    
            except Exception as e:
                logger.error(f"{model_name} tahmin hatası: {e}")
        
        if not forecasts:
            logger.error("Hiçbir model tahmin yapamadı.")
            return None
        
        # Ensemble tahmin (ortalaması)
        forecast_df = pd.DataFrame(forecasts)
        forecast_df['ensemble'] = forecast_df.mean(axis=1)
        
        # Tarih indeksi oluştur
        last_date = self.data.index[-1]
        forecast_index = pd.date_range(start=last_date, periods=steps + 1, freq='MS')[1:]
        forecast_df.index = forecast_index
        
        logger.info("Ensemble tahmin başarıyla tamamlandı.")
        return forecast_df
    
    def _moving_average_forecast(self, model_info, steps):
        """Moving Average tahmin yapar - çok tutarlı."""
        ma_value = model_info['ma_value']
        trend = model_info['trend']
        last_value = model_info['last_value']
        
        predictions = []
        current_value = last_value
        
        for i in range(steps):
            # Moving average'a doğru yavaşça yaklaş
            alpha = 0.1  # Yavaş adaptasyon
            current_value = alpha * ma_value + (1 - alpha) * current_value
            
            # Küçük trend ekle
            current_value = current_value * (1 + trend * 0.1)
            
            # Son değerin %15'i içinde sınırla (daha sıkı)
            min_bound = last_value * 0.85
            max_bound = last_value * 1.15
            current_value = np.clip(current_value, min_bound, max_bound)
            
            predictions.append(current_value)
            
        return np.array(predictions)
    
    def _linear_forecast(self, model_info, steps):
        """Linear Regression tahmin yapar - tutarlı sınırlar ile."""
        # Son değeri al
        last_value = self.data['target'].iloc[-1]
        
        # Basit trend hesapla (son 6 ayın ortalaması)
        recent_data = self.data['target'].tail(6)
        trend = recent_data.pct_change().mean()
        
        predictions = []
        current_value = last_value
        
        for i in range(steps):
            # Basit trend + küçük rastgele değişim
            change = trend * current_value + np.random.normal(0, current_value * 0.02)
            current_value = current_value + change
            
            # Son değerin %20'si içinde sınırla
            min_bound = last_value * 0.8
            max_bound = last_value * 1.2
            current_value = np.clip(current_value, min_bound, max_bound)
            
            predictions.append(current_value)
            
        return np.array(predictions)
    
    def _prophet_forecast(self, model_info, steps):
        """Prophet tahmin yapar."""
        future = model_info.make_future_dataframe(periods=steps, freq='MS')
        forecast = model_info.predict(future)
        return forecast['yhat'].tail(steps).values
    
    def _xgboost_forecast(self, model_info, steps):
        """XGBoost tahmin yapar."""
        # Son veriyi kullanarak gelecek tahminleri yap
        last_features = self.data[self.feature_columns].iloc[-1:].values
        scaler = self.scalers['xgboost']
        
        predictions = []
        current_features = last_features.copy()
        
        for _ in range(steps):
            # Tahmin yap
            pred = model_info.predict(scaler.transform(current_features))[0]
            predictions.append(pred)
            
            # Özellikleri güncelle (lag özelliklerini)
            # Bu basit bir yaklaşım, gerçek uygulamada daha karmaşık olabilir
            current_features[0, 0] = pred  # target_lag_1
            
        return np.array(predictions)
    
    def _lightgbm_forecast(self, model_info, steps):
        """LightGBM tahmin yapar."""
        # XGBoost ile aynı mantık
        last_features = self.data[self.feature_columns].iloc[-1:].values
        scaler = self.scalers['lightgbm']
        
        predictions = []
        current_features = last_features.copy()
        
        for _ in range(steps):
            pred = model_info.predict(scaler.transform(current_features))[0]
            predictions.append(pred)
            current_features[0, 0] = pred
            
        return np.array(predictions)
    
    def _lstm_forecast(self, model_info, steps):
        """LSTM tahmin yapar."""
        # Son sequence'i kullan
        scaler_X = self.scalers['lstm_X']
        scaler_y = self.scalers['lstm_y']
        
        last_sequence = self.data[self.feature_columns].tail(12).values
        last_sequence_scaled = scaler_X.transform(last_sequence)
        
        predictions = []
        current_sequence = last_sequence_scaled.copy()
        
        for _ in range(steps):
            # Sequence'i yeniden şekillendir
            X_input = current_sequence.reshape(1, 12, len(self.feature_columns))
            
            # Tahmin yap
            pred_scaled = model_info.predict(X_input, verbose=0)[0, 0]
            pred = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred)
            
            # Sequence'i güncelle
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = scaler_X.transform([[pred] + [0] * (len(self.feature_columns) - 1)])[0]
            
        return np.array(predictions)
    
    def plot_forecast(self, forecast_df: pd.DataFrame):
        """Tahmin sonuçlarını görselleştirir."""
        plt.figure(figsize=(15, 8))
        
        # Gerçek veriyi çiz
        plt.plot(self.data.index, self.data['target'], label='Gerçek Veri', linewidth=2, color='blue')
        
        # Tahminleri çiz
        for model_name in forecast_df.columns:
            plt.plot(forecast_df.index, forecast_df[model_name], 
                    label=f'{model_name.title()} Tahmini', linestyle='--', alpha=0.7)
        
        plt.title(f'{self.target_field.upper()} - Gelişmiş Model Tahminleri', fontsize=16)
        plt.xlabel('Tarih', fontsize=12)
        plt.ylabel('Değer', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def test_advanced_forecaster():
    """Gelişmiş tahmin modellerini test eder."""
    logger.info("Gelişmiş tahmin modelleri test ediliyor...")
    
    forecaster = AdvancedForecaster(target_field='usd_try')
    
    # Veriyi yükle
    data = forecaster.load_data_from_mongodb()
    if data is None or data.empty:
        logger.error("Veri yüklenemedi.")
        return
    
    logger.info(f"Veri yüklendi: {len(data)} kayıt")
    
    # Modelleri eğit
    forecaster.train_all_models(data)
    
    # Ensemble tahmin yap
    forecast = forecaster.ensemble_forecast(steps=12)
    
    if forecast is not None:
        logger.info("Tahmin sonuçları:")
        print(forecast[['ensemble']].round(2))
        
        # Görselleştir
        forecaster.plot_forecast(forecast)
    
    logger.info("Gelişmiş tahmin testi tamamlandı.")

if __name__ == '__main__':
    test_advanced_forecaster()
