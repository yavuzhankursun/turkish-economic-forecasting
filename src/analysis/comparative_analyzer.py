"""
Karşılaştırmalı Analiz Modülü
==============================

TÜBİTAK Projesi - Farklı ekonomik göstergelerin karşılaştırmalı analizi

G-11 Gereksinimi: Sistem, farklı ekonomik göstergelerin karşılaştırmalı analizini yapabilen bir modüle sahip olmalıdır.
G-12 Gereksinimi: Karşılaştırmalı analiz sistemi, en az 3 farklı karşılaştırma metriği kullanmalıdır.

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComparativeAnalyzer:
    """
    Farklı ekonomik göstergelerin karşılaştırmalı analizini yapan sınıf.
    
    G-11 ve G-12 gereksinimleri için:
    - Korelasyon analizi
    - Trend benzerliği analizi
    - Volatilite karşılaştırması
    """
    
    def __init__(self):
        """ComparativeAnalyzer'ı başlatır."""
        logger.info("ComparativeAnalyzer başlatıldı.")
    
    def calculate_correlation(self, series1: pd.Series, series2: pd.Series, 
                             method: str = 'pearson') -> Dict[str, float]:
        """
        İki zaman serisi arasındaki korelasyonu hesaplar.
        
        G-12 Metrik 1: Korelasyon analizi
        
        Args:
            series1: İlk zaman serisi
            series2: İkinci zaman serisi
            method: Korelasyon yöntemi ('pearson', 'spearman')
            
        Returns:
            Dict[str, float]: Korelasyon metrikleri
        """
        # İndeksleri hizala
        aligned = pd.DataFrame({'series1': series1, 'series2': series2}).dropna()
        
        if len(aligned) < 3:
            logger.warning("Yetersiz veri için korelasyon hesaplanamadı")
            return {'correlation': 0.0, 'p_value': 1.0, 'method': method}
        
        if method == 'pearson':
            corr, p_value = pearsonr(aligned['series1'], aligned['series2'])
        elif method == 'spearman':
            corr, p_value = spearmanr(aligned['series1'], aligned['series2'])
        else:
            corr = aligned['series1'].corr(aligned['series2'])
            p_value = np.nan
        
        result = {
            'correlation': float(corr),
            'p_value': float(p_value) if not np.isnan(p_value) else None,
            'method': method,
            'n_samples': len(aligned)
        }
        
        logger.info(f"Korelasyon ({method}): {corr:.4f} (p={p_value:.4f if p_value else 'N/A'})")
        return result
    
    def calculate_trend_similarity(self, series1: pd.Series, series2: pd.Series) -> Dict[str, float]:
        """
        İki zaman serisinin trend benzerliğini hesaplar.
        
        G-12 Metrik 2: Trend benzerliği analizi
        
        Args:
            series1: İlk zaman serisi
            series2: İkinci zaman serisi
            
        Returns:
            Dict[str, float]: Trend benzerliği metrikleri
        """
        # İndeksleri hizala
        aligned = pd.DataFrame({'series1': series1, 'series2': series2}).dropna()
        
        if len(aligned) < 3:
            logger.warning("Yetersiz veri için trend benzerliği hesaplanamadı")
            return {'trend_similarity': 0.0, 'direction_match': 0.0}
        
        # Farkları hesapla (değişim yönü)
        diff1 = aligned['series1'].diff().dropna()
        diff2 = aligned['series2'].diff().dropna()
        
        # Ortak indeksleri bul
        common_idx = diff1.index.intersection(diff2.index)
        if len(common_idx) < 2:
            return {'trend_similarity': 0.0, 'direction_match': 0.0}
        
        diff1_aligned = diff1.loc[common_idx]
        diff2_aligned = diff2.loc[common_idx]
        
        # Yön eşleşmesi (aynı yönde hareket eden yüzdesi)
        direction_match = ((diff1_aligned > 0) == (diff2_aligned > 0)).mean()
        
        # Trend korelasyonu (farklar arası korelasyon)
        trend_corr = diff1_aligned.corr(diff2_aligned)
        if np.isnan(trend_corr):
            trend_corr = 0.0
        
        # Trend benzerliği skoru (0-1 arası)
        trend_similarity = (direction_match + (trend_corr + 1) / 2) / 2
        
        result = {
            'trend_similarity': float(trend_similarity),
            'direction_match': float(direction_match),
            'trend_correlation': float(trend_corr),
            'n_samples': len(common_idx)
        }
        
        logger.info(f"Trend benzerliği: {trend_similarity:.4f} (yön eşleşmesi: {direction_match:.2%})")
        return result
    
    def calculate_volatility_comparison(self, series1: pd.Series, series2: pd.Series) -> Dict[str, float]:
        """
        İki zaman serisinin volatilitesini karşılaştırır.
        
        G-12 Metrik 3: Volatilite karşılaştırması
        
        Args:
            series1: İlk zaman serisi
            series2: İkinci zaman serisi
            
        Returns:
            Dict[str, float]: Volatilite karşılaştırma metrikleri
        """
        # İndeksleri hizala
        aligned = pd.DataFrame({'series1': series1, 'series2': series2}).dropna()
        
        if len(aligned) < 3:
            logger.warning("Yetersiz veri için volatilite karşılaştırması hesaplanamadı")
            return {'volatility_ratio': 1.0, 'cv1': 0.0, 'cv2': 0.0}
        
        # Yüzde değişimleri hesapla
        pct_change1 = aligned['series1'].pct_change().dropna()
        pct_change2 = aligned['series2'].pct_change().dropna()
        
        # Ortak indeksleri bul
        common_idx = pct_change1.index.intersection(pct_change2.index)
        if len(common_idx) < 2:
            return {'volatility_ratio': 1.0, 'cv1': 0.0, 'cv2': 0.0}
        
        pct_change1_aligned = pct_change1.loc[common_idx]
        pct_change2_aligned = pct_change2.loc[common_idx]
        
        # Standart sapma (volatilite)
        std1 = pct_change1_aligned.std()
        std2 = pct_change2_aligned.std()
        
        # Coefficient of Variation (CV)
        mean1 = abs(pct_change1_aligned.mean())
        mean2 = abs(pct_change2_aligned.mean())
        cv1 = std1 / (mean1 + 1e-10)
        cv2 = std2 / (mean2 + 1e-10)
        
        # Volatilite oranı
        volatility_ratio = std1 / (std2 + 1e-10)
        
        result = {
            'volatility_ratio': float(volatility_ratio),
            'std1': float(std1),
            'std2': float(std2),
            'cv1': float(cv1),
            'cv2': float(cv2),
            'n_samples': len(common_idx)
        }
        
        logger.info(f"Volatilite karşılaştırması: Oran={volatility_ratio:.4f} (CV1={cv1:.4f}, CV2={cv2:.4f})")
        return result
    
    def compare_indicators(self, indicators: Dict[str, pd.Series]) -> Dict[str, Dict]:
        """
        Birden fazla göstergeyi karşılaştırır.
        
        G-11 gereksinimi: Karşılaştırmalı analiz modülü
        
        Args:
            indicators: Gösterge adı -> zaman serisi dict'i
            
        Returns:
            Dict[str, Dict]: Karşılaştırma sonuçları
        """
        indicator_names = list(indicators.keys())
        
        if len(indicator_names) < 2:
            logger.warning("En az 2 gösterge gereklidir")
            return {}
        
        results = {}
        
        # Tüm çiftleri karşılaştır
        for i, name1 in enumerate(indicator_names):
            for name2 in indicator_names[i+1:]:
                series1 = indicators[name1]
                series2 = indicators[name2]
                
                comparison_key = f"{name1}_vs_{name2}"
                
                # G-12: En az 3 karşılaştırma metriği
                comparison_results = {
                    'correlation': self.calculate_correlation(series1, series2),
                    'trend_similarity': self.calculate_trend_similarity(series1, series2),
                    'volatility_comparison': self.calculate_volatility_comparison(series1, series2)
                }
                
                results[comparison_key] = comparison_results
                
                logger.info(f"Karşılaştırma tamamlandı: {name1} vs {name2}")
        
        logger.info(f"Toplam {len(results)} karşılaştırma yapıldı")
        return results
    
    def generate_comparison_report(self, comparison_results: Dict[str, Dict]) -> str:
        """
        Karşılaştırma sonuçlarını rapor olarak üretir.
        
        Args:
            comparison_results: compare_indicators() sonuçları
            
        Returns:
            str: Rapor metni
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("KARŞILAŞTIRMALI ANALİZ RAPORU")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        for comparison_key, metrics in comparison_results.items():
            report_lines.append(f"Karşılaştırma: {comparison_key}")
            report_lines.append("-" * 80)
            
            # Korelasyon
            corr = metrics.get('correlation', {})
            report_lines.append(f"Korelasyon ({corr.get('method', 'pearson')}): {corr.get('correlation', 0):.4f}")
            if corr.get('p_value'):
                report_lines.append(f"  P-değeri: {corr.get('p_value', 0):.4f}")
            
            # Trend benzerliği
            trend = metrics.get('trend_similarity', {})
            report_lines.append(f"Trend Benzerliği: {trend.get('trend_similarity', 0):.4f}")
            report_lines.append(f"  Yön Eşleşmesi: {trend.get('direction_match', 0):.2%}")
            
            # Volatilite
            vol = metrics.get('volatility_comparison', {})
            report_lines.append(f"Volatilite Oranı: {vol.get('volatility_ratio', 1):.4f}")
            report_lines.append(f"  CV1: {vol.get('cv1', 0):.4f}, CV2: {vol.get('cv2', 0):.4f}")
            
            report_lines.append("")
        
        report_lines.append("=" * 80)
        return "\n".join(report_lines)


def compare_economic_indicators(indicators: Dict[str, pd.Series]) -> Dict[str, Dict]:
    """
    Ekonomik göstergeleri karşılaştırmak için kolay kullanım fonksiyonu.
    
    Args:
        indicators: Gösterge adı -> zaman serisi dict'i
        
    Returns:
        Dict[str, Dict]: Karşılaştırma sonuçları
    """
    analyzer = ComparativeAnalyzer()
    return analyzer.compare_indicators(indicators)


if __name__ == '__main__':
    # Test
    logger.info("ComparativeAnalyzer test ediliyor...")
    
    # Örnek veriler
    dates = pd.date_range('2024-01-01', periods=100, freq='MS')
    usd_try = pd.Series(np.random.randn(100).cumsum() + 30, index=dates)
    inflation = pd.Series(np.random.randn(100).cumsum() + 50, index=dates)
    interest_rate = pd.Series(np.random.randn(100).cumsum() + 25, index=dates)
    
    indicators = {
        'usd_try': usd_try,
        'inflation': inflation,
        'interest_rate': interest_rate
    }
    
    analyzer = ComparativeAnalyzer()
    results = analyzer.compare_indicators(indicators)
    
    report = analyzer.generate_comparison_report(results)
    logger.info("\n" + report)
    
    logger.info("ComparativeAnalyzer test tamamlandı.")

