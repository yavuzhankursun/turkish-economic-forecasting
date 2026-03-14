"""
Validation Analiz Aracı
=======================

MongoDB'deki TimeSeriesSplit validation sonuçlarını analiz eder ve görselleştirir.
Politika faizi ve TÜFE'deki kopuklukları ve yüksek hata oranlarını tespit eder.

Geliştirici: [Öğrenci Adı]
Tarih: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from typing import Dict, List, Optional
import sys
import os

# Proje kök dizinini path'e ekle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.mongodb_manager import MongoDBManager
from config.config import MONGODB_COLLECTIONS

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Matplotlib Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class ValidationAnalyzer:
    """
    MongoDB'deki validation sonuçlarını analiz eden sınıf.
    """
    
    def __init__(self):
        """ValidationAnalyzer'ı başlat."""
        self.db_manager = None
        
    def connect(self):
        """MongoDB bağlantısını kur."""
        try:
            self.db_manager = MongoDBManager()
            if self.db_manager.client is None:
                logger.error("MongoDB bağlantısı kurulamadı!")
                return False
            logger.info("✅ MongoDB bağlantısı başarılı!")
            return True
        except Exception as e:
            logger.error(f"MongoDB bağlantı hatası: {e}")
            return False
    
    def get_validation_history(self, indicator_type: str = None, limit: int = 100) -> pd.DataFrame:
        """
        Validation geçmişini MongoDB'den çeker.
        
        Args:
            indicator_type: Gösterge tipi ('usd_try', 'inflation', 'interest_rate') veya None (hepsi)
            limit: Maksimum kayıt sayısı
            
        Returns:
            pd.DataFrame: Validation geçmişi
        """
        if self.db_manager is None:
            logger.error("MongoDB bağlantısı yok!")
            return pd.DataFrame()
        
        try:
            query = {}
            if indicator_type:
                # Hem indicator_type hem de indicator field'larını kontrol et
                query['$or'] = [
                    {'indicator_type': indicator_type},
                    {'indicator': indicator_type}
                ]
            
            collection = self.db_manager.get_collection('timeseries_validation')
            if collection is None:
                logger.error("timeseries_validation koleksiyonu bulunamadı!")
                return pd.DataFrame()
            
            # En son kayıtları getir
            cursor = collection.find(query).sort('timestamp', -1).limit(limit)
            records = list(cursor)
            
            if not records:
                logger.warning("Validation kaydı bulunamadı!")
                return pd.DataFrame()
            
            # DataFrame'e çevir
            df = pd.DataFrame(records)
            
            # Timestamp'i datetime'a çevir
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"✅ {len(df)} validation kaydı yüklendi.")
            return df
            
        except Exception as e:
            logger.error(f"Validation geçmişi yükleme hatası: {e}")
            return pd.DataFrame()
    
    def analyze_errors_by_date(self, indicator_type: str) -> pd.DataFrame:
        """
        Tarihe göre hata analizi yapar.
        
        Args:
            indicator_type: Gösterge tipi
            
        Returns:
            pd.DataFrame: Tarih bazlı hata analizi
        """
        df = self.get_validation_history(indicator_type=indicator_type)
        
        if df.empty:
            return pd.DataFrame()
        
        # Fold bazlı analiz
        error_analysis = []
        
        for _, row in df.iterrows():
            # Hem fold_metrics hem de folds field'larını kontrol et
            fold_data = None
            if 'fold_metrics' in row and isinstance(row['fold_metrics'], list):
                fold_data = row['fold_metrics']
            elif 'folds' in row and isinstance(row['folds'], list):
                fold_data = row['folds']
            
            if fold_data:
                for fold_idx, fold_metric in enumerate(fold_data):
                    if isinstance(fold_metric, dict):
                        error_analysis.append({
                            'timestamp': row.get('timestamp', pd.NaT),
                            'indicator_type': row.get('indicator_type', row.get('indicator', indicator_type)),
                            'fold': fold_metric.get('fold', fold_idx + 1),
                            'rmse': fold_metric.get('rmse', 0),
                            'mape': fold_metric.get('mape', 0),
                            'mae': fold_metric.get('mae', 0),
                            'test_size': fold_metric.get('test_size', 0),
                            'train_size': fold_metric.get('train_size', 0),
                            'samples': fold_metric.get('samples', 0)
                        })
        
        if not error_analysis:
            logger.warning("Fold metrikleri bulunamadı!")
            return pd.DataFrame()
        
        error_df = pd.DataFrame(error_analysis)
        return error_df
    
    def find_high_error_periods(self, indicator_type: str, mape_threshold: float = 50.0) -> pd.DataFrame:
        """
        Yüksek hata oranına sahip dönemleri bulur.
        
        Args:
            indicator_type: Gösterge tipi
            mape_threshold: MAPE eşik değeri (%)
            
        Returns:
            pd.DataFrame: Yüksek hatalı dönemler
        """
        error_df = self.analyze_errors_by_date(indicator_type)
        
        if error_df.empty:
            return pd.DataFrame()
        
        # Yüksek hatalı dönemleri filtrele
        high_error = error_df[error_df['mape'] > mape_threshold].copy()
        
        if not high_error.empty:
            logger.info(f"⚠️ {len(high_error)} yüksek hatalı dönem bulundu (MAPE > {mape_threshold}%)")
            # Tarihe göre sırala
            high_error = high_error.sort_values('timestamp')
        else:
            logger.info(f"✅ Tüm dönemlerde MAPE < {mape_threshold}%")
        
        return high_error
    
    def plot_error_trends(self, indicator_type: str, save_path: str = None):
        """
        Hata trendlerini görselleştirir.
        
        Args:
            indicator_type: Gösterge tipi
            save_path: Kayıt yolu (None ise gösterir)
        """
        error_df = self.analyze_errors_by_date(indicator_type)
        
        if error_df.empty:
            logger.warning("Görselleştirme için veri yok!")
            return
        
        # Tarihe göre grupla ve ortalama al
        if 'timestamp' in error_df.columns:
            error_df['date'] = pd.to_datetime(error_df['timestamp']).dt.date
            daily_avg = error_df.groupby('date').agg({
                'rmse': 'mean',
                'mape': 'mean',
                'mae': 'mean'
            }).reset_index()
            
            # Grafik oluştur
            fig, axes = plt.subplots(3, 1, figsize=(14, 10))
            fig.suptitle(f'{indicator_type.upper()} - Validation Hata Trendleri', fontsize=16, fontweight='bold')
            
            # RMSE
            axes[0].plot(daily_avg['date'], daily_avg['rmse'], marker='o', color='blue', linewidth=2)
            axes[0].set_title('RMSE Trendi', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('RMSE', fontsize=10)
            axes[0].grid(True, alpha=0.3)
            axes[0].tick_params(axis='x', rotation=45)
            
            # MAPE
            axes[1].plot(daily_avg['date'], daily_avg['mape'], marker='s', color='red', linewidth=2)
            axes[1].axhline(y=50, color='orange', linestyle='--', label='Uyarı Eşiği (50%)')
            axes[1].set_title('MAPE Trendi (%)', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('MAPE (%)', fontsize=10)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].tick_params(axis='x', rotation=45)
            
            # MAE
            axes[2].plot(daily_avg['date'], daily_avg['mae'], marker='^', color='green', linewidth=2)
            axes[2].set_title('MAE Trendi', fontsize=12, fontweight='bold')
            axes[2].set_ylabel('MAE', fontsize=10)
            axes[2].set_xlabel('Tarih', fontsize=10)
            axes[2].grid(True, alpha=0.3)
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"✅ Grafik kaydedildi: {save_path}")
            else:
                plt.show()
        else:
            logger.warning("Timestamp bilgisi yok, görselleştirme yapılamıyor!")
    
    def generate_report(self, indicator_type: str = None) -> str:
        """
        Validation analiz raporu oluşturur.
        
        Args:
            indicator_type: Gösterge tipi veya None (hepsi)
            
        Returns:
            str: Rapor metni
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("VALIDATION ANALİZ RAPORU")
        report_lines.append("=" * 80)
        report_lines.append(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        indicators = ['usd_try', 'inflation', 'interest_rate'] if indicator_type is None else [indicator_type]
        
        for ind in indicators:
            report_lines.append(f"\n{'=' * 80}")
            report_lines.append(f"GÖSTERGE: {ind.upper()}")
            report_lines.append(f"{'=' * 80}")
            
            error_df = self.analyze_errors_by_date(ind)
            
            if error_df.empty:
                report_lines.append("⚠️ Veri bulunamadı!")
                continue
            
            # Genel istatistikler
            report_lines.append("\n📊 GENEL İSTATİSTİKLER:")
            report_lines.append(f"  Toplam Validation: {len(error_df)}")
            report_lines.append(f"  Ortalama RMSE: {error_df['rmse'].mean():.4f}")
            report_lines.append(f"  Ortalama MAPE: {error_df['mape'].mean():.2f}%")
            report_lines.append(f"  Ortalama MAE: {error_df['mae'].mean():.4f}")
            report_lines.append(f"  Maksimum MAPE: {error_df['mape'].max():.2f}%")
            report_lines.append(f"  Minimum MAPE: {error_df['mape'].min():.2f}%")
            
            # Yüksek hatalı dönemler
            high_error = self.find_high_error_periods(ind, mape_threshold=50.0)
            if not high_error.empty:
                report_lines.append(f"\n⚠️ YÜKSEK HATALI DÖNEMLER (MAPE > 50%):")
                report_lines.append(f"  Toplam: {len(high_error)} dönem")
                for _, row in high_error.head(10).iterrows():
                    report_lines.append(
                        f"  - {row.get('timestamp', 'N/A')}: MAPE={row['mape']:.2f}%, "
                        f"RMSE={row['rmse']:.4f}, Fold={row['fold']}"
                    )
            else:
                report_lines.append("\n✅ Yüksek hatalı dönem bulunamadı (MAPE < 50%)")
        
        report_lines.append("\n" + "=" * 80)
        report = "\n".join(report_lines)
        
        return report
    
    def close(self):
        """Bağlantıyı kapat."""
        if self.db_manager:
            self.db_manager.close_connection()


def main():
    """Ana fonksiyon - komut satırından çalıştırma."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validation Analiz Aracı')
    parser.add_argument('--indicator', type=str, choices=['usd_try', 'inflation', 'interest_rate'],
                       help='Analiz edilecek gösterge')
    parser.add_argument('--plot', action='store_true', help='Grafik oluştur')
    parser.add_argument('--save-plot', type=str, help='Grafik kayıt yolu')
    parser.add_argument('--report', action='store_true', help='Rapor oluştur')
    parser.add_argument('--save-report', type=str, help='Rapor kayıt yolu')
    parser.add_argument('--high-error', action='store_true', help='Yüksek hatalı dönemleri göster')
    
    args = parser.parse_args()
    
    analyzer = ValidationAnalyzer()
    
    if not analyzer.connect():
        logger.error("MongoDB bağlantısı kurulamadı!")
        return
    
    try:
        # Rapor oluştur
        if args.report:
            report = analyzer.generate_report(indicator_type=args.indicator)
            print(report)
            
            if args.save_report:
                with open(args.save_report, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"✅ Rapor kaydedildi: {args.save_report}")
        
        # Grafik oluştur
        if args.plot and args.indicator:
            analyzer.plot_error_trends(args.indicator, save_path=args.save_plot)
        
        # Yüksek hatalı dönemler
        if args.high_error and args.indicator:
            high_error = analyzer.find_high_error_periods(args.indicator)
            if not high_error.empty:
                print(f"\n⚠️ Yüksek Hatalı Dönemler ({args.indicator}):")
                print(high_error.to_string())
            else:
                print(f"\n✅ Yüksek hatalı dönem bulunamadı ({args.indicator})")
        
        # Varsayılan: Rapor göster
        if not args.report and not args.plot and not args.high_error:
            report = analyzer.generate_report(indicator_type=args.indicator)
            print(report)
            
    finally:
        analyzer.close()


if __name__ == '__main__':
    main()

