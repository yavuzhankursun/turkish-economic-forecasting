# --- START OF FILE turkish_financial_scraper.py ---

import requests
from bs4 import BeautifulSoup
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import time
from io import StringIO
import re
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.align import Align
from rich import box
from rich.text import Text
import warnings

# --- Proje Kök Dizinini Path'e Ekleme ---
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
# -----------------------------------------

# Yeni eklenen importlar

from src.utils.mongodb_manager import MongoDBManager
from config.config import TABLE_NAMES, MONGODB_COLLECTIONS

warnings.filterwarnings('ignore')

class TurkishFinancialScraper:
    def __init__(self):
        self.console = Console()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
        })

    def print_header(self, title):
        """Başlık yazdırma fonksiyonu"""
        panel = Panel(
            Align.center(Text(title, style="bold cyan")),
            box=box.DOUBLE_EDGE,
            style="bright_blue"
        )
        self.console.print(panel)
        self.console.print()

    def get_historical_usd_try(self, start_year=2020):
        """Geçmiş USD/TRY verilerini toplu çek"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("USD/TRY historical veriler çekiliyor...", total=None)

            usd_data = []
            current_date = datetime.now()
            start_date = datetime(start_year, 1, 1)

            # Her ayın son iş günü için veri çek
            while start_date <= current_date:
                # Ayın son günü
                if start_date.month == 12:
                    next_month = start_date.replace(year=start_date.year + 1, month=1, day=1)
                else:
                    next_month = start_date.replace(month=start_date.month + 1, day=1)

                last_day_of_month = next_month - timedelta(days=1)

                # Hafta sonu ise geriye git
                while last_day_of_month.weekday() >= 5: # Cumartesi (5) veya Pazar (6)
                    last_day_of_month -= timedelta(days=1)

                try:
                    year_month_str = last_day_of_month.strftime("%Y%m")
                    date_str = last_day_of_month.strftime("%d%m%Y")
                    url = f"https://www.tcmb.gov.tr/kurlar/{year_month_str}/{date_str}.xml"

                    response = self.session.get(url, timeout=10)
                    response.raise_for_status()

                    root = ET.fromstring(response.content)

                    for currency in root.findall('Currency'):
                        if currency.get('Kod') == 'USD':
                            buying = float(currency.find('BanknoteBuying').text)
                            selling = float(currency.find('BanknoteSelling').text)
                            avg_rate = (buying + selling) / 2

                            usd_data.append({
                                'year': last_day_of_month.year,
                                'month': last_day_of_month.month,
                                'date': last_day_of_month.strftime('%Y-%m'),
                                'usd_avg': avg_rate
                            })
                            break

                    time.sleep(0.1)  # Rate limiting

                except Exception as e:
                    # Veri bulunamazsa bir önceki ayın verisini kullan (forward fill)
                    if usd_data:
                        last_data = usd_data[-1].copy()
                        last_data.update({
                            'year': last_day_of_month.year,
                            'month': last_day_of_month.month,
                            'date': last_day_of_month.strftime('%Y-%m')
                        })
                        usd_data.append(last_data)
                        self.console.print(f"[yellow]⚠️ {last_day_of_month.strftime('%Y-%m')} USD verisi bulunamadı, önceki değer kullanıldı[/yellow]")

                start_date = next_month

            progress.update(task, completed=True)
            if not usd_data:
                self.console.print("[red]❌ Hiç USD verisi çekilemedi.[/red]")
                return pd.DataFrame()
            return pd.DataFrame(usd_data)

    def get_historical_tufe(self):
        """TCMB'den TÜFE historical verilerini çek (Geliştirilmiş ve Düzeltilmiş Versiyon)"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("TÜFE historical veriler çekiliyor...", total=None)
            
            url = "https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Istatistikler/Enflasyon+Verileri/Tuketici+Fiyatlari"
            
            try:
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                
                tables = pd.read_html(StringIO(response.text))
                
                tufe_table = None
                # Strateji 1: Kolon adlarında "TÜFE" ve "Yıllık" kelimelerini ara. Bu en sağlam yöntem.
                for table in tables:
                    if any("TÜFE" in str(col) and "Yıllık" in str(col) for col in table.columns):
                        tufe_table = table
                        break
                
                # Strateji 2 (Yedek): Eğer ilk strateji başarısız olursa, ilk hücresi "AY-YIL" formatında olan tabloyu ara.
                if tufe_table is None:
                    for table in tables:
                        if not table.empty and table.shape[1] >= 2 and re.match(r'\d{2}-\d{4}', str(table.iloc[0, 0])):
                            tufe_table = table
                            break
                
                if tufe_table is not None:
                    # Kolon isimlerini temizle ve standardize et
                    tufe_table.columns = ['date_str', 'tufe_rate'] + list(tufe_table.columns[2:])
                    
                    # Sadece gerekli kolonları al
                    df = tufe_table[['date_str', 'tufe_rate']].copy()
                    
                    # Tarih formatını standardize et (MM-YYYY -> YYYY-MM)
                    df['date'] = df['date_str'].apply(self.parse_date_string)
                    df = df.dropna(subset=['date'])
                    
                    # Yıl ve ay kolonları ekle
                    df['year'] = df['date'].apply(lambda x: int(x.split('-')[0]))
                    df['month'] = df['date'].apply(lambda x: int(x.split('-')[1]))
                    
                    # Numerik olmayan değerleri temizle
                    df['tufe_rate'] = pd.to_numeric(df['tufe_rate'], errors='coerce')
                    df = df.dropna(subset=['tufe_rate'])
                    
                    progress.update(task, completed=True)
                    return df[['year', 'month', 'date', 'tufe_rate']].sort_values(['year', 'month'])
                else:
                    self.console.print("[red]❌ TÜFE verisi içeren tablo web sayfasında bulunamadı.[/red]")

            except Exception as e:
                self.console.print(f"[red]❌ TÜFE verisi çekilemedi: {e}[/red]")
            
            progress.update(task, completed=True)
            return None

    def get_historical_repo_rates(self):
        """TCMB'den repo faiz oranları historical verilerini çek"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Repo faiz historical veriler çekiliyor...", total=None)
            
            url = "https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Temel+Faaliyetler/Para+Politikasi/Merkez+Bankasi+Faiz+Oranlari/1+Hafta+Repo"
            
            try:
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                
                tables = pd.read_html(StringIO(response.text))
                
                if len(tables) > 0:
                    repo_table = tables[0]
                    
                    if not repo_table.empty and repo_table.shape[1] >= 3:
                        repo_table = repo_table.iloc[:, :3] # Sadece ilk 3 kolonu al
                        repo_table.columns = ["date_str", "borrow_rate", "lend_rate"]
                        
                        # Tarih formatını datetime'a çevir
                        repo_table['date_dt'] = pd.to_datetime(repo_table['date_str'], format='%d.%m.%Y', errors='coerce')
                        repo_table = repo_table.dropna(subset=['date_dt'])
                        
                        # Faiz oranlarını numeric'e çevir (virgülü noktaya çevir)
                        repo_table['lend_rate'] = pd.to_numeric(repo_table['lend_rate'].astype(str).str.replace(',', '.'), errors='coerce')
                        
                        # Yıl-ay bazında grupla ve son değeri al (aylık son faiz oranı)
                        repo_table['year'] = repo_table['date_dt'].dt.year
                        repo_table['month'] = repo_table['date_dt'].dt.month
                        
                        # Her ay için son kaydı al
                        monthly_repo = repo_table.sort_values('date_dt').groupby(['year', 'month']).last().reset_index()
                        monthly_repo['date'] = monthly_repo['date_dt'].dt.strftime('%Y-%m')

                        progress.update(task, completed=True)
                        return monthly_repo[['year', 'month', 'date', 'lend_rate']]
                        
            except Exception as e:
                self.console.print(f"[red]❌ Repo faiz verisi çekilemedi: {e}[/red]")
            
            progress.update(task, completed=True)
            return None

    def parse_date_string(self, date_str):
        """Tarih string'ini YYYY-MM formatına çevir"""
        try:
            if pd.isna(date_str):
                return None
            
            date_str = str(date_str).strip()
            
            # MM-YYYY formatı
            match = re.match(r'(\d{2})-(\d{4})', date_str)
            if match:
                month, year = match.groups()
                return f"{year}-{month.zfill(2)}"
            
            # YYYY-MM formatı (zaten doğru)
            if re.match(r'\d{4}-\d{2}', date_str):
                return date_str
                
            return None
        except:
            return None

    def synchronize_data(self, usd_df, tufe_df, repo_df):
        """3 veri setini senkronize et ve eksik değerleri doldur"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Veriler senkronize ediliyor...", total=None)
            
            # Tüm veri setlerinden en geniş tarih aralığını bul
            all_dates = set()
            
            if usd_df is not None and not usd_df.empty:
                all_dates.update(usd_df[['year', 'month']].apply(lambda x: (x['year'], x['month']), axis=1))
            
            if tufe_df is not None and not tufe_df.empty:
                all_dates.update(tufe_df[['year', 'month']].apply(lambda x: (x['year'], x['month']), axis=1))
            
            if repo_df is not None and not repo_df.empty:
                all_dates.update(repo_df[['year', 'month']].apply(lambda x: (x['year'], x['month']), axis=1))
            
            if not all_dates:
                self.console.print("[red]❌ Senkronizasyon için hiçbir veri bulunamadı. İşlem durduruldu.[/red]")
                progress.update(task, completed=True)
                return pd.DataFrame()
            
            # En geniş tarih aralığından base DataFrame oluştur
            sorted_dates = sorted(all_dates)
            base_data = []
            for year, month in sorted_dates:
                base_data.append({
                    'year': year,
                    'month': month,
                    'date': f"{year}-{month:02d}"
                })
            
            base_df = pd.DataFrame(base_data)

            # USD verilerini merge et
            if usd_df is not None and not usd_df.empty:
                base_df = pd.merge(base_df, usd_df[['year', 'month', 'usd_avg']], on=['year', 'month'], how='left')
            else:
                base_df['usd_avg'] = None

            # TÜFE verilerini merge et
            if tufe_df is not None:
                base_df = pd.merge(base_df, tufe_df[['year', 'month', 'tufe_rate']], on=['year', 'month'], how='left')
            else:
                base_df['tufe_rate'] = None

            # Repo faiz verilerini merge et
            if repo_df is not None and not repo_df.empty:
                base_df = pd.merge(base_df, repo_df[['year', 'month', 'lend_rate']], on=['year', 'month'], how='left')
            else:
                base_df['lend_rate'] = None
            
            # Sadece tüm değişkenlerin mevcut olduğu satırları tut
            numeric_columns = ['usd_avg', 'tufe_rate', 'lend_rate']
            available_columns = [col for col in numeric_columns if col in base_df.columns]
            
            if available_columns:
                # Tüm değişkenlerin mevcut olduğu satırları filtrele
                base_df = base_df.dropna(subset=available_columns)
                
                if base_df.empty:
                    self.console.print("[red]❌ Tüm değişkenlerin mevcut olduğu veri bulunamadı.[/red]")
                    progress.update(task, completed=True)
                    return pd.DataFrame()
                
                self.console.print(f"[green]✅ {len(base_df)} satır veri (tüm değişkenler mevcut) senkronize edildi.[/green]")
            else:
                self.console.print("[red]❌ Hiçbir sayısal değişken bulunamadı.[/red]")
                progress.update(task, completed=True)
                return pd.DataFrame()

            progress.update(task, completed=True)
            return base_df.sort_values(['year', 'month']).reset_index(drop=True)

    def create_synchronized_table(self, sync_df, show_last_n=15):
        """Senkronize veri için güzel tablo oluştur"""
        table = Table(
            title="📊 ARIMA Modeli için Senkronize Veri Seti (Son 15 Ay)",
            box=box.ROUNDED,
            header_style="bold magenta"
        )
        
        table.add_column("Tarih", style="cyan", no_wrap=True)
        table.add_column("USD/TRY (Ort.)", style="green", justify="right")
        table.add_column("TÜFE (Yıllık %)", style="blue", justify="right")
        table.add_column("Repo Faizi (%)", style="red", justify="right")
        
        # Son N kaydı göster
        recent_data = sync_df.tail(show_last_n)
        
        for _, row in recent_data.iterrows():
            table.add_row(
                row['date'],
                f"{row.get('usd_avg', 0):.4f}",
                f"{row.get('tufe_rate', 0):.2f}",
                f"{row.get('lend_rate', 0):.2f}"
            )
        
        return table

    def get_arima_ready_data(self, start_year=2020):
        """ARIMA modeli için hazır veri seti döndür - Güncel tarihe kadar"""
        self.print_header("🤖 ARIMA Modeli için Veri Hazırlığı")
        
        # Güncel tarihe kadar veri çek
        current_year = datetime.now().year
        self.console.print(f"[cyan]📅 Veri çekme aralığı: {start_year} - {current_year}[/cyan]")
        
        usd_df = self.get_historical_usd_try(start_year)
        tufe_df = self.get_historical_tufe()
        repo_df = self.get_historical_repo_rates()
        
        # Eğer temel veri setlerinden biri çekilemediyse işlemi durdur
        if usd_df is None or tufe_df is None or repo_df is None:
             self.console.print("[bold red]❌ Gerekli temel verilerden biri veya birkaçı çekilemedi. Senkronizasyon yapılamıyor.[/bold red]")
             return pd.DataFrame()

        self.console.print("\n[bold]🔄 Veri Senkronizasyonu ve Sonuçlar[/bold]")
        synchronized_df = self.synchronize_data(usd_df, tufe_df, repo_df)
        
        if not synchronized_df.empty:
            table = self.create_synchronized_table(synchronized_df)
            self.console.print(table)
            
            total_records = len(synchronized_df)
            start_date = synchronized_df.iloc[0]['date']
            end_date = synchronized_df.iloc[-1]['date']
            
            summary_panel = Panel(
                f"[bold green]✅ Veri senkronizasyonu tamamlandı![/bold green]\n\n"
                f"📊 Toplam Kayıt: {total_records}\n"
                f"📅 Tarih Aralığı: {start_date} - {end_date}\n"
                f"🎯 ARIMA modeli için hazır!",
                title="📈 Veri Seti Özeti",
                box=box.ROUNDED
            )
            self.console.print(summary_panel)
        
        return synchronized_df

    def save_data_to_db(self, synchronized_df):
        """Senkronize veriyi hem PostgreSQL hem de MongoDB'ye kaydeder."""
        if synchronized_df.empty:
            self.console.print("[yellow]⚠️ Kaydedilecek veri bulunmadığı için veritabanı işlemi yapılmadı.[/yellow]")
            return False
        
        success_count = 0
        total_attempts = 0
        
        # PostgreSQL'e kaydetme kısmı kaldırıldı (Sadece MongoDB kullanılıyor)
        # try:
        #     self.console.print("\n[bold]💾 PostgreSQL'e Kaydetme[/bold]")
        #     total_attempts += 1
        #     db_manager = DatabaseManager(env='production')
        #     
        #     if db_manager.engine is not None:
        #         table_name = TABLE_NAMES.get('economic_data', 'financial_data_raw')
        #         db_manager.save_dataframe(synchronized_df, table_name=table_name, if_exists='replace')
        #         self.console.print(f"[green]✅ Veri başarıyla PostgreSQL [bold]{table_name}[/bold] tablosuna kaydedildi![/green]")
        #         success_count += 1
        #     else:
        #         self.console.print("[yellow]⚠️ PostgreSQL bağlantısı kurulamadı, MongoDB'ye kaydediliyor.[/yellow]")
        #         
        # except Exception as e:
        #     self.console.print(f"[yellow]⚠️ PostgreSQL'e kaydetme hatası: {e}[/yellow]")
        
        # MongoDB'ye kaydetme
        try:
            self.console.print("\n[bold]💾 MongoDB'ye Kaydetme[/bold]")
            total_attempts += 1
            mongodb_manager = MongoDBManager()
            
            if mongodb_manager.client is not None and mongodb_manager.database is not None:
                collection_name = MONGODB_COLLECTIONS.get('economic_indicators', 'economic_indicators')
                
                # DataFrame'i standart MongoDB belgelerine dönüştür
                documents = []
                current_time = datetime.utcnow()
                
                for _, row in synchronized_df.iterrows():
                    # Standart veri yapısı oluştur
                    doc = {
                        'date': row.get('date', current_time.date()),
                        'created_at': current_time,
                        'updated_at': current_time,
                        'source': 'TurkishFinancialScraper',
                        'data_type': 'economic_indicators'
                    }
                    
                    # Scraping modülünden gelen verileri standart alanlara eşleştir
                    # USD/TRY kuru
                    if 'usd_avg' in row and pd.notna(row['usd_avg']):
                        doc['usd_try'] = float(row['usd_avg'])
                    elif 'usd_try' in row and pd.notna(row['usd_try']):
                        doc['usd_try'] = float(row['usd_try'])
                    
                    # EUR/TRY kuru (varsa)
                    if 'eur_try' in row and pd.notna(row['eur_try']):
                        doc['eur_try'] = float(row['eur_try'])
                    
                    # Faiz oranları
                    if 'lend_rate' in row and pd.notna(row['lend_rate']):
                        doc['policy_rate'] = float(row['lend_rate'])
                    elif 'policy_rate' in row and pd.notna(row['policy_rate']):
                        doc['policy_rate'] = float(row['policy_rate'])
                    
                    # Enflasyon oranı
                    if 'tufe_rate' in row and pd.notna(row['tufe_rate']):
                        doc['inflation_rate'] = float(row['tufe_rate'])
                    
                    # Tarih bilgileri
                    if 'year' in row and pd.notna(row['year']):
                        doc['year'] = int(row['year'])
                    if 'month' in row and pd.notna(row['month']):
                        doc['month'] = int(row['month'])
                    
                    # Diğer alanları koru
                    for col in synchronized_df.columns:
                        if col not in ['date', 'created_at', 'updated_at', 'source']:
                            if col not in doc:  # Eğer standart eşleştirme yapılmadıysa
                                doc[col] = row[col]
                    
                    documents.append(doc)
                
                # Belgeleri ekle
                inserted_ids = mongodb_manager.insert_many_documents(collection_name, documents)
                if inserted_ids:
                    self.console.print(f"[green]✅ Veri başarıyla MongoDB [bold]{collection_name}[/bold] koleksiyonuna kaydedildi! ({len(documents)} belge)[/green]")
                    success_count += 1
                else:
                    self.console.print("[red]❌ MongoDB'ye veri eklenemedi.[/red]")
                
                # Bağlantıyı kapat
                mongodb_manager.close_connection()
            else:
                self.console.print("[yellow]⚠️ MongoDB bağlantısı kurulamadı.[/yellow]")
                
        except Exception as e:
            self.console.print(f"[red]❌ MongoDB'ye kaydetme hatası: {e}[/red]")
        
        # Sonuç özeti
        if success_count > 0:
            self.console.print(f"\n[green]🎉 Toplam {success_count}/{total_attempts} veritabanına başarıyla kaydedildi![/green]")
            return True
        else:
            self.console.print(f"\n[red]❌ Hiçbir veritabanına kaydedilemedi![/red]")
            return False

def run_scraper():
    """Scraper'ı çalıştıran ve tüm adımları yürüten ana fonksiyon."""
    scraper = TurkishFinancialScraper()
    
    # Adım 1: ARIMA için veriyi hazırla
    scraper.print_header("🤖 ARIMA Modeli için Veri Hazırlığı")
    synchronized_df = scraper.get_arima_ready_data()
    
    if synchronized_df is None or synchronized_df.empty:
        scraper.console.print("[bold red]❌ Senkronize edilecek veri bulunamadı. İşlem durduruldu.[/bold red]")
        return
        
    # Adım 2: Veritabanına kaydet
    scraper.print_header("💾 Veritabanına Kaydetme")
    scraper.save_data_to_db(synchronized_df)

# Betik doğrudan çalıştırıldığında...
if __name__ == '__main__':
    run_scraper()