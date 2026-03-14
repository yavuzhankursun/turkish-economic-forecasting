"""
Veri Ön İşleme Modülü
======================

TÜBİTAK Projesi - Eksik değer interpolasyonu, aykırı değer tespiti, veri normalizasyonu

G-6 Gereksinimi: Veri ön işleme süreci; eksik değerlerin tespiti ve interpolasyonu, 
aykırı değerlerin belirlenmesi ve filtrelenmesi, zaman serisi formatına dönüştürme 
ve veri normalizasyonu adımlarını içermelidir.

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Veri ön işleme sınıfı.
    
    Özellikler:
    - Eksik değer tespiti ve interpolasyonu
    - Aykırı değer tespiti ve filtreleme
    - Zaman serisi formatına dönüştürme
    - Veri normalizasyonu
    """
    
    def __init__(self):
        """DataPreprocessor'ı başlatır."""
        logger.info("DataPreprocessor başlatıldı.")
    
    def detect_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Eksik değerleri tespit eder.
        
        Args:
            df: İncelenecek DataFrame
            
        Returns:
            Dict[str, int]: Sütun bazlı eksik değer sayıları
        """
        missing_counts = df.isnull().sum().to_dict()
        missing_percentages = {col: (count / len(df)) * 100 
                             for col, count in missing_counts.items() 
                             if count > 0}
        
        logger.info(f"Eksik değer tespit edildi: {len(missing_percentages)} sütun")
        for col, pct in missing_percentages.items():
            logger.info(f"  {col}: {missing_counts[col]} eksik ({pct:.2f}%)")
        
        return missing_counts
    
    def interpolate_missing_values(self, df: pd.DataFrame, 
                                   method: str = 'time',
                                   limit: Optional[int] = None) -> pd.DataFrame:
        """
        Eksik değerleri interpolasyon ile doldurur.
        
        Args:
            df: İşlenecek DataFrame
            method: İnterpolasyon yöntemi ('time', 'linear', 'polynomial', 'spline')
            limit: Maksimum interpolasyon sayısı
            
        Returns:
            pd.DataFrame: İnterpolasyon yapılmış DataFrame
        """
        df = df.copy()
        
        # Zaman serisi için time-based interpolasyon
        if method == 'time' and isinstance(df.index, pd.DatetimeIndex):
            # Zaman bazlı interpolasyon
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].isnull().any():
                    original_count = df[col].isnull().sum()
                    df[col] = df[col].interpolate(method='time', limit=limit)
                    filled_count = original_count - df[col].isnull().sum()
                    if filled_count > 0:
                        logger.info(f"  {col}: {filled_count} eksik değer interpolasyon ile dolduruldu")
        else:
            # Lineer interpolasyon
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].isnull().any():
                    original_count = df[col].isnull().sum()
                    df[col] = df[col].interpolate(method='linear', limit=limit)
                    filled_count = original_count - df[col].isnull().sum()
                    if filled_count > 0:
                        logger.info(f"  {col}: {filled_count} eksik değer interpolasyon ile dolduruldu")
        
        logger.info("Eksik değer interpolasyonu tamamlandı")
        return df
    
    def detect_outliers(self, df: pd.DataFrame, 
                       method: str = 'iqr',
                       threshold: float = 3.0) -> Dict[str, List[int]]:
        """
        Aykırı değerleri tespit eder.
        
        Args:
            df: İncelenecek DataFrame
            method: Tespit yöntemi ('iqr', 'zscore', 'modified_zscore')
            threshold: Eşik değeri
            
        Returns:
            Dict[str, List[int]]: Sütun bazlı aykırı değer indeksleri
        """
        outliers = {}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            values = df[col].dropna()
            
            if len(values) < 3:
                continue
            
            if method == 'iqr':
                # IQR (Interquartile Range) yöntemi
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            
            elif method == 'zscore':
                # Z-score yöntemi
                z_scores = np.abs(stats.zscore(values))
                outlier_indices = df[z_scores > threshold].index.tolist()
            
            elif method == 'modified_zscore':
                # Modified Z-score yöntemi (median kullanır)
                median = values.median()
                mad = np.median(np.abs(values - median))  # Median Absolute Deviation
                if mad > 0:
                    modified_z_scores = 0.6745 * (values - median) / mad
                    outlier_indices = df[np.abs(modified_z_scores) > threshold].index.tolist()
                else:
                    outlier_indices = []
            
            else:
                outlier_indices = []
            
            if outlier_indices:
                outliers[col] = outlier_indices
                logger.info(f"  {col}: {len(outlier_indices)} aykırı değer tespit edildi")
        
        logger.info(f"Aykırı değer tespiti tamamlandı: {len(outliers)} sütun")
        return outliers
    
    def filter_outliers(self, df: pd.DataFrame, 
                       outliers: Dict[str, List[int]],
                       method: str = 'remove') -> pd.DataFrame:
        """
        Aykırı değerleri filtreler.
        
        Args:
            df: İşlenecek DataFrame
            outliers: Tespit edilen aykırı değerler
            method: Filtreleme yöntemi ('remove', 'clip', 'median')
            
        Returns:
            pd.DataFrame: Filtrelenmiş DataFrame
        """
        df = df.copy()
        
        for col, indices in outliers.items():
            if col not in df.columns:
                continue
            
            if method == 'remove':
                # Aykırı değerleri kaldır
                df.loc[indices, col] = np.nan
                logger.info(f"  {col}: {len(indices)} aykırı değer kaldırıldı")
            
            elif method == 'clip':
                # Aykırı değerleri sınırlara çek
                values = df[col].dropna()
                if len(values) > 0:
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
                    logger.info(f"  {col}: Aykırı değerler sınırlara çekildi")
            
            elif method == 'median':
                # Aykırı değerleri medyan ile değiştir
                median_value = df[col].median()
                df.loc[indices, col] = median_value
                logger.info(f"  {col}: {len(indices)} aykırı değer medyan ile değiştirildi")
        
        logger.info("Aykırı değer filtreleme tamamlandı")
        return df
    
    def convert_to_time_series(self, df: pd.DataFrame, 
                              date_column: Optional[str] = None,
                              freq: Optional[str] = None) -> pd.DataFrame:
        """
        DataFrame'i zaman serisi formatına dönüştürür.
        
        Args:
            df: Dönüştürülecek DataFrame
            date_column: Tarih sütunu adı (None ise index kullanılır)
            freq: Zaman serisi frekansı ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            pd.DataFrame: Zaman serisi formatında DataFrame
        """
        df = df.copy()
        
        # Tarih sütununu ayarla
        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.set_index(date_column)
        elif not isinstance(df.index, pd.DatetimeIndex):
            # Index'i tarih formatına çevirmeyi dene
            try:
                df.index = pd.to_datetime(df.index)
            except:
                logger.warning("Index tarih formatına çevrilemedi")
        
        # Frekans ayarla
        if freq:
            try:
                df = df.asfreq(freq, method='pad')
                logger.info(f"Zaman serisi frekansı ayarlandı: {freq}")
            except Exception as e:
                logger.warning(f"Frekans ayarlanamadı: {e}")
        
        # Sırala
        df = df.sort_index()
        
        logger.info(f"Zaman serisi formatına dönüştürüldü: {len(df)} satır")
        return df
    
    def normalize_data(self, df: pd.DataFrame, 
                     method: str = 'standard',
                     columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Veriyi normalleştirir.
        
        Args:
            df: Normalleştirilecek DataFrame
            method: Normalleştirme yöntemi ('standard', 'minmax', 'robust')
            columns: Normalleştirilecek sütunlar (None ise tüm sayısal sütunlar)
            
        Returns:
            pd.DataFrame: Normalleştirilmiş DataFrame
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            logger.warning("Normalleştirilecek sayısal sütun bulunamadı")
            return df
        
        # Normalleştirme yöntemini seç
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            logger.warning(f"Bilinmeyen normalleştirme yöntemi: {method}, standard kullanılıyor")
            scaler = StandardScaler()
        
        # Normalleştir
        for col in columns:
            if col in df.columns:
                values = df[col].values.reshape(-1, 1)
                normalized = scaler.fit_transform(values)
                df[col] = normalized.flatten()
                logger.info(f"  {col}: {method} normalleştirme uygulandı")
        
        logger.info(f"Veri normalleştirme tamamlandı: {len(columns)} sütun")
        return df
    
    def preprocess_pipeline(self, df: pd.DataFrame,
                           interpolate: bool = True,
                           detect_outliers: bool = True,
                           filter_outliers: bool = True,
                           normalize: bool = False,
                           convert_to_ts: bool = True) -> pd.DataFrame:
        """
        Tam veri ön işleme pipeline'ı.
        
        Args:
            df: İşlenecek DataFrame
            interpolate: İnterpolasyon yapılsın mı
            detect_outliers: Aykırı değer tespiti yapılsın mı
            filter_outliers: Aykırı değer filtreleme yapılsın mı
            normalize: Normalleştirme yapılsın mı
            convert_to_ts: Zaman serisi formatına dönüştürülsün mü
            
        Returns:
            pd.DataFrame: Ön işleme yapılmış DataFrame
        """
        logger.info("Veri ön işleme pipeline'ı başlatıldı")
        
        # 1. Eksik değer tespiti
        missing = self.detect_missing_values(df)
        
        # 2. İnterpolasyon
        if interpolate:
            df = self.interpolate_missing_values(df, method='time')
        
        # 3. Aykırı değer tespiti
        outliers = {}
        if detect_outliers:
            outliers = self.detect_outliers(df, method='iqr')
        
        # 4. Aykırı değer filtreleme
        if filter_outliers and outliers:
            df = self.filter_outliers(df, outliers, method='median')
            # Filtreleme sonrası tekrar interpolasyon
            if interpolate:
                df = self.interpolate_missing_values(df, method='time')
        
        # 5. Zaman serisi formatına dönüştürme
        if convert_to_ts:
            df = self.convert_to_time_series(df)
        
        # 6. Normalleştirme (opsiyonel)
        if normalize:
            df = self.normalize_data(df, method='standard')
        
        logger.info("Veri ön işleme pipeline'ı tamamlandı")
        return df


def preprocess_dataframe(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    DataFrame'i ön işlemek için kolay kullanım fonksiyonu.
    
    Args:
        df: İşlenecek DataFrame
        **kwargs: DataPreprocessor.preprocess_pipeline() parametreleri
        
    Returns:
        pd.DataFrame: Ön işleme yapılmış DataFrame
    """
    preprocessor = DataPreprocessor()
    return preprocessor.preprocess_pipeline(df, **kwargs)


if __name__ == '__main__':
    # Test
    logger.info("DataPreprocessor test ediliyor...")
    preprocessor = DataPreprocessor()
    
    # Örnek veri oluştur
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    data = {
        'value': np.random.randn(100).cumsum() + 100,
        'value2': np.random.randn(100).cumsum() + 50
    }
    df = pd.DataFrame(data, index=dates)
    
    # Bazı eksik değerler ekle
    df.loc[df.index[10:15], 'value'] = np.nan
    
    # Ön işleme yap
    processed_df = preprocessor.preprocess_pipeline(df)
    
    logger.info(f"Ön işleme tamamlandı: {len(processed_df)} satır")
    logger.info("DataPreprocessor test tamamlandı.")

