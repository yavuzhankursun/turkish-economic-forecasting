"""
Rapor Export Modülü
===================

TÜBİTAK Projesi - PNG, SVG, PDF formatlarında rapor export

G-16 Gereksinimi: Sistem, analiz sonuçlarını ve görselleri otomatik olarak 
PNG, SVG ve PDF formatlarında raporlayabilmelidir.

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib.backends.backend_svg import FigureCanvasSVG
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path
import sys
import os

# Proje kök dizinini import yolu olarak ekle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportExporter:
    """
    Raporları farklı formatlarda export eden sınıf.
    
    Desteklenen formatlar:
    - PNG
    - SVG
    - PDF
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        ReportExporter'ı başlatır.
        
        Args:
            output_dir: Export dizini (None ise mevcut dizin)
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / 'reports'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ReportExporter başlatıldı: {self.output_dir}")
    
    def export_figure_to_png(self, fig, filename: str, dpi: int = 300) -> str:
        """
        Matplotlib figürünü PNG formatında kaydeder.
        
        Args:
            fig: Matplotlib figure
            filename: Dosya adı
            dpi: Çözünürlük
            
        Returns:
            str: Kaydedilen dosya yolu
        """
        filepath = self.output_dir / f"{filename}.png"
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', format='png')
        logger.info(f"PNG export tamamlandı: {filepath}")
        return str(filepath)
    
    def export_figure_to_svg(self, fig, filename: str) -> str:
        """
        Matplotlib figürünü SVG formatında kaydeder.
        
        Args:
            fig: Matplotlib figure
            filename: Dosya adı
            
        Returns:
            str: Kaydedilen dosya yolu
        """
        filepath = self.output_dir / f"{filename}.svg"
        fig.savefig(filepath, format='svg', bbox_inches='tight')
        logger.info(f"SVG export tamamlandı: {filepath}")
        return str(filepath)
    
    def create_analysis_pdf(self, analysis_data: Dict, filename: str, 
                           language: str = 'tr') -> str:
        """
        Analiz sonuçlarını PDF formatında rapor olarak oluşturur.
        
        Args:
            analysis_data: Analiz verileri
            filename: Dosya adı
            language: Dil ('tr' veya 'en')
            
        Returns:
            str: Kaydedilen dosya yolu
        """
        filepath = self.output_dir / f"{filename}.pdf"
        
        # PDF dokümanı oluştur
        doc = SimpleDocTemplate(str(filepath), pagesize=A4)
        story = []
        
        # Stil tanımlamaları
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1E40AF'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        # Başlık
        if language == 'tr':
            title_text = "TÜBİTAK Ekonomik Göstergeler Tahmin Raporu"
        else:
            title_text = "TÜBİTAK Economic Indicators Forecast Report"
        
        story.append(Paragraph(title_text, title_style))
        story.append(Spacer(1, 0.2 * inch))
        
        # Tarih
        date_text = datetime.now().strftime('%d.%m.%Y %H:%M')
        story.append(Paragraph(f"<b>Tarih / Date:</b> {date_text}", styles['Normal']))
        story.append(Spacer(1, 0.3 * inch))
        
        # Göstergeler
        indicators = ['usd_try', 'inflation', 'interest_rate']
        indicator_names_tr = {
            'usd_try': 'USD/TRY Kuru',
            'inflation': 'Enflasyon Oranı',
            'interest_rate': 'Faiz Oranı'
        }
        indicator_names_en = {
            'usd_try': 'USD/TRY Exchange Rate',
            'inflation': 'Inflation Rate',
            'interest_rate': 'Interest Rate'
        }
        
        indicator_names = indicator_names_tr if language == 'tr' else indicator_names_en
        
        for indicator_key in indicators:
            if indicator_key not in analysis_data:
                continue
            
            indicator_data = analysis_data[indicator_key]
            if indicator_data.get('status') != 'success':
                continue
            
            # Gösterge başlığı
            story.append(Paragraph(f"<b>{indicator_names[indicator_key]}</b>", styles['Heading2']))
            story.append(Spacer(1, 0.1 * inch))
            
            # Metrikler tablosu
            metrics = indicator_data.get('metrics', {})
            if metrics:
                table_data = []
                if language == 'tr':
                    table_data.append(['Metrik', 'Değer'])
                    table_data.append(['RMSE', f"{metrics.get('RMSE', 0):.4f}"])
                    table_data.append(['MAE', f"{metrics.get('MAE', 0):.4f}"])
                    table_data.append(['MAPE', f"{metrics.get('MAPE', 0):.2f}%"])
                else:
                    table_data.append(['Metric', 'Value'])
                    table_data.append(['RMSE', f"{metrics.get('RMSE', 0):.4f}"])
                    table_data.append(['MAE', f"{metrics.get('MAE', 0):.4f}"])
                    table_data.append(['MAPE', f"{metrics.get('MAPE', 0):.2f}%"])
                
                table = Table(table_data, colWidths=[2 * inch, 2 * inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
                story.append(Spacer(1, 0.2 * inch))
            
            story.append(PageBreak())
        
        # PDF'i oluştur
        doc.build(story)
        logger.info(f"PDF export tamamlandı: {filepath}")
        return str(filepath)
    
    def export_all_formats(self, fig, analysis_data: Dict, base_filename: str,
                          language: str = 'tr') -> Dict[str, str]:
        """
        Tüm formatlarda export yapar.
        
        Args:
            fig: Matplotlib figure
            analysis_data: Analiz verileri
            base_filename: Temel dosya adı
            language: Dil
            
        Returns:
            Dict[str, str]: Export edilen dosya yolları
        """
        results = {}
        
        # PNG export
        results['png'] = self.export_figure_to_png(fig, base_filename)
        
        # SVG export
        results['svg'] = self.export_figure_to_svg(fig, base_filename)
        
        # PDF export
        results['pdf'] = self.create_analysis_pdf(analysis_data, base_filename, language)
        
        logger.info(f"Tüm formatlarda export tamamlandı: {base_filename}")
        return results


def export_report(fig, analysis_data: Dict, base_filename: str, 
                 output_dir: Optional[str] = None, language: str = 'tr') -> Dict[str, str]:
    """
    Raporu tüm formatlarda export etmek için kolay kullanım fonksiyonu.
    
    Args:
        fig: Matplotlib figure
        analysis_data: Analiz verileri
        base_filename: Temel dosya adı
        output_dir: Export dizini
        language: Dil
        
    Returns:
        Dict[str, str]: Export edilen dosya yolları
    """
    exporter = ReportExporter(output_dir)
    return exporter.export_all_formats(fig, analysis_data, base_filename, language)


if __name__ == '__main__':
    # Test
    logger.info("ReportExporter test ediliyor...")
    
    # Örnek figür
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
    ax.set_title('Test Grafik')
    
    # Örnek analiz verisi
    analysis_data = {
        'usd_try': {
            'status': 'success',
            'metrics': {'RMSE': 0.5, 'MAE': 0.3, 'MAPE': 2.5}
        }
    }
    
    exporter = ReportExporter()
    results = exporter.export_all_formats(fig, analysis_data, 'test_report', 'tr')
    
    logger.info(f"Export sonuçları: {results}")
    logger.info("ReportExporter test tamamlandı.")

