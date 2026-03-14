
"""
TÜBİTAK Ekonomik Göstergeler Tahmin Projesi - Web Uygulaması
=============================================================

Flask backend API servisi.

Geliştirici: [Öğrenci Adı]
Danışman: [Danışman Adı]
"""

from flask import Flask, jsonify, request, send_from_directory, abort
from flask_cors import CORS
from datetime import datetime
import logging
import sys
import os
import traceback
import pandas as pd

# Proje kök dizinini path'e ekle
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import modülleri
try:
    from src.models.arima_model import ARIMAForecaster, load_complete_data_from_mongodb
    from src.models.inflation_forecaster import InflationForecaster, load_inflation_data_from_mongodb
    from src.models.interest_rate_forecaster import InterestRateForecaster, load_interest_rate_data_from_mongodb
    from src.models.model_validator import ModelValidator, validate_arima_model
    from src.services.news_api_service import NewsAPIService, ClaudeAPIService
    from src.services.multi_indicator_service import MultiIndicatorNewsService
    from src.systems.autonomous_forecaster import AutonomousForecastingSystem
    from src.core.economic_analyzer import EconomicAnalyzer
    from src.utils.mongodb_manager import MongoDBManager
    from config.config import NEWS_API_CONFIG, CLAUDE_API_CONFIG
except ImportError as e:
    print(f"HATA: Modül yüklenemedi: {e}")
    sys.exit(1)

# Flask uygulaması
app = Flask(__name__)
CORS(app)  # CORS desteği

# Numpy tiplerini Python native tiplerine dönüştür (JSON serialization için)
# Bu fonksiyon NumpyJSONProvider'dan önce tanımlanmalı
def convert_numpy_types(obj):
    """Numpy tiplerini Python native tiplerine dönüştürür - recursive ve kapsamlı."""
    import numpy as np
    
    # Numpy tiplerini kontrol et - önce numpy tiplerini kontrol et
    if isinstance(obj, np.bool_) or type(obj).__name__ == 'bool_':
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                        np.uint8, np.uint16, np.uint32, np.uint64)) or 'int' in str(type(obj)):
        try:
            return int(obj)
        except:
            return obj
    elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)) or 'float' in str(type(obj)):
        try:
            return float(obj)
        except:
            return obj
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_types(item) for item in obj.tolist()]
    # Pandas tipleri
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
    elif isinstance(obj, pd.Series):
        return [convert_numpy_types(item) for item in obj.tolist()]
    elif isinstance(obj, pd.DataFrame):
        return {str(key): convert_numpy_types(value) for key, value in obj.to_dict(orient='index').items()}
    # Recursive dönüştürme - dict, list, tuple, set
    elif isinstance(obj, dict):
        return {str(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, set):
        return [convert_numpy_types(item) for item in obj]
    # Diğer tipler için type name kontrolü (fallback)
    elif hasattr(obj, '__class__') and 'numpy' in str(type(obj)):
        type_name = type(obj).__name__
        if 'bool' in type_name.lower():
            return bool(obj)
        elif 'int' in type_name.lower():
            try:
                return int(obj)
            except:
                return obj
        elif 'float' in type_name.lower():
            try:
                return float(obj)
            except:
                return obj
    return obj

# Flask JSON encoder'ını özelleştir - numpy tiplerini otomatik dönüştür
from flask.json.provider import DefaultJSONProvider
import json

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        import numpy as np
        # Numpy tiplerini kontrol et
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        # Recursive olarak dict ve list içindeki numpy tiplerini dönüştür
        elif isinstance(obj, dict):
            return {key: self.default(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.default(item) for item in obj]
        return super().default(obj)
    
    def dumps(self, obj, **kwargs):
        # Önce convert_numpy_types ile dönüştür, sonra JSON'a çevir
        import json
        converted_obj = convert_numpy_types(obj)
        # super().dumps() yerine doğrudan json.dumps kullan (recursive çağrıyı önlemek için)
        return json.dumps(converted_obj, **kwargs)

app.json = NumpyJSONProvider(app)

FRONTEND_DIST_DIR = os.path.join(project_root, 'frontend', 'dist')

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hata yakalama decorator'ı
def handle_errors(f):
    """API endpoint'leri için hata yakalama decorator'ı."""
    def wrapper(*args, **kwargs):
        try:
            result = f(*args, **kwargs)
            # Eğer result bir Response objesi ise ve içinde numpy tipleri varsa dönüştür
            # (jsonify zaten bunu yapıyor ama ekstra güvenlik için)
            return result
        except Exception as e:
            logger.error(f"API Hatası ({f.__name__}): {e}")
            logger.error(traceback.format_exc())
            error_msg = str(e)
            # Eğer hata mesajında numpy tipi varsa dönüştür
            try:
                error_msg = convert_numpy_types(error_msg) if isinstance(error_msg, (dict, list)) else error_msg
            except:
                pass
            return jsonify({
                'success': False,
                'message': error_msg if isinstance(error_msg, str) else 'Beklenmeyen hata: ' + str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    wrapper.__name__ = f.__name__
    return wrapper

# ============================================================================
# WEB SAYFALARI
# ============================================================================

def _frontend_available() -> bool:
    return os.path.exists(os.path.join(FRONTEND_DIST_DIR, 'index.html'))


@app.route('/')
def index():
    return serve_frontend('index.html')


@app.route('/data')
def data_page():
    return serve_frontend('index.html')


@app.route('/multi')
def multi_page():
    return serve_frontend('index.html')

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
@handle_errors
def health_check():
    """Sistem sağlık kontrolü."""
    logger.info("Sağlık kontrolü yapılıyor...")
    
    health_status = {
        'mongodb': False,
        'news_api': False,
        'claude_api': False,
        'arima_model': False
    }
    
    messages = []
    
    # MongoDB kontrolü
    try:
        with MongoDBManager() as mongo_manager:
            if mongo_manager.client is not None:
                collection = mongo_manager.database['economic_indicators']
                count = collection.count_documents({})
                health_status['mongodb'] = True
                messages.append(f"MongoDB: OK ({count} kayıt)")
            else:
                messages.append("MongoDB: Bağlantı yok")
    except Exception as e:
        messages.append(f"MongoDB: Hata - {str(e)}")
    
    # News API kontrolü
    try:
        news_key = NEWS_API_CONFIG.get("api_key", "")
        if news_key and news_key != "YOUR_NEWS_API_KEY_HERE":
            health_status['news_api'] = True
            messages.append("News API: OK")
        else:
            messages.append("News API: Anahtar eksik")
    except Exception as e:
        messages.append(f"News API: Hata - {str(e)}")
    
    # Claude API kontrolü
    try:
        claude_key = CLAUDE_API_CONFIG.get("api_key", "")
        if claude_key and claude_key.startswith("sk-ant-"):
            health_status['claude_api'] = True
            messages.append("Claude API: OK")
        else:
            messages.append("Claude API: Anahtar eksik/geçersiz")
    except Exception as e:
        messages.append(f"Claude API: Hata - {str(e)}")
    
    # ARIMA model kontrolü
    try:
        data = load_complete_data_from_mongodb(target_field='usd_try')
        if data is not None and not data.empty:
            health_status['arima_model'] = True
            messages.append(f"ARIMA Model: OK ({len(data)} veri noktası)")
        else:
            messages.append("ARIMA Model: Veri yok")
    except Exception as e:
        messages.append(f"ARIMA Model: Hata - {str(e)}")
    
    overall_health = all(health_status.values())
    
    return jsonify({
        'success': True,
        'data': {
            'status': health_status,
            'messages': messages,
            'overall_health': overall_health
        },
        'message': 'Sağlık kontrolü tamamlandı',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/data/latest', methods=['GET'])
@handle_errors
def get_latest_data():
    """En son verileri getirir."""
    logger.info("En son veriler getiriliyor...")
    
    data = load_complete_data_from_mongodb(target_field='usd_try')
    
    if data is None or data.empty:
        return jsonify({
            'success': False,
            'message': 'Veri bulunamadı',
            'timestamp': datetime.now().isoformat()
        }), 404
    
    # Son 20 kayıt
    latest_data = data.tail(20)
    
    # JSON formatına çevir
    data_list = []
    for date, value in latest_data.items():
        data_list.append({
            'date': date.strftime('%Y-%m-%d'),
            'value': float(value)
        })
    
    return jsonify({
        'success': True,
        'data': {
            'records': data_list,
            'count': len(data_list),
            'latest_value': float(latest_data.iloc[-1]),
            'latest_date': latest_data.index[-1].strftime('%Y-%m-%d')
        },
        'message': 'Veriler başarıyla getirildi',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/arima/forecast', methods=['POST'])
@handle_errors
def arima_forecast():
    """ARIMA tahmin yapar."""
    logger.info("ARIMA tahmin yapılıyor...")
    
    # İstek parametreleri
    params = request.get_json() or {}
    steps = params.get('steps', 12)
    test_size = params.get('test_size', 0.2)
    
    # Veri yükle - USD/TRY için özel yükleme
    from src.models.arima_model import load_data_from_mongodb
    data = load_data_from_mongodb(target_field='usd_try')
    
    if data is None or data.empty:
        logger.warning("Alternatif veri yükleme deneniyor...")
        data = load_complete_data_from_mongodb(target_field='usd_try')
    
    if data is None or data.empty:
        return jsonify({
            'success': False,
            'message': 'Veri bulunamadı',
            'timestamp': datetime.now().isoformat()
        }), 404
    
    # ARIMA modeli
    forecaster = ARIMAForecaster(target_column='usd_try')
    forecaster.fit(data, test_size=test_size)
    
    # Performans değerlendirmesi
    try:
        metrics = forecaster.evaluate()
    except Exception as e:
        logger.warning(f"Metrik hesaplama hatası: {e}")
        metrics = None
    
    # Tahmin
    forecast_results = forecaster.forecast(steps=steps)
    
    if forecast_results is None:
        return jsonify({
            'success': False,
            'message': 'Tahmin yapılamadı',
            'timestamp': datetime.now().isoformat()
        }), 500
    
    # Tarihsel veri
    historical = []
    for date, value in data.items():
        historical.append({
            'date': date.strftime('%Y-%m-%d'),
            'value': float(value)
        })
    
    # Tahmin verileri
    forecast = []
    for date, value in forecast_results['forecast'].items():
        forecast.append({
            'date': date.strftime('%Y-%m-%d'),
            'value': float(value)
        })
    
    return jsonify({
        'success': True,
        'data': {
            'historical': historical,
            'forecast': forecast,
            'metrics': metrics,
            'model_params': forecaster.best_params
        },
        'message': 'ARIMA tahmini başarıyla tamamlandı',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model/validate', methods=['POST'])
@handle_errors
def validate_model():
    """Model validasyonu yapar (hold-out ve cross-validation)."""
    logger.info("Model validasyonu yapılıyor...")
    
    params = request.get_json() or {}
    indicator_type = params.get('indicator_type', 'usd_try')  # 'usd_try', 'inflation', 'interest_rate'
    validation_type = params.get('validation_type', 'both')  # 'holdout', 'crossval', 'both'
    test_size = params.get('test_size', 0.2)
    n_splits = params.get('n_splits', 5)
    
    # Enflasyon için özel validasyon parametreleri (daha az volatilite için)
    if indicator_type == 'inflation':
        # Enflasyon verisi volatil olduğu için daha az fold ve daha büyük test setleri kullan
        if n_splits > 3:
            logger.info(f"Enflasyon için fold sayısı {n_splits}'den 3'e düşürülüyor (volatilite nedeniyle)")
            n_splits = 3
        if test_size < 0.25:
            logger.info(f"Enflasyon için test_size {test_size}'dan 0.25'e artırılıyor")
            test_size = 0.25
    
    # Veri yükle
    data = None
    forecaster_class = None
    
    if indicator_type == 'usd_try':
        from src.models.arima_model import load_data_from_mongodb
        data = load_data_from_mongodb(target_field='usd_try')
        if data is None or data.empty:
            data = load_complete_data_from_mongodb(target_field='usd_try')
        forecaster_class = ARIMAForecaster
    elif indicator_type == 'inflation':
        data = load_inflation_data_from_mongodb()
        forecaster_class = InflationForecaster
    elif indicator_type == 'interest_rate':
        data = load_interest_rate_data_from_mongodb()
        forecaster_class = InterestRateForecaster
    else:
        return jsonify({
            'success': False,
            'message': f'Geçersiz gösterge tipi: {indicator_type}',
            'timestamp': datetime.now().isoformat()
        }), 400
    
    if data is None or data.empty:
        return jsonify({
            'success': False,
            'message': f'{indicator_type} verisi bulunamadı. MongoDB bağlantısını ve veri aktarımını kontrol edin.',
            'timestamp': datetime.now().isoformat()
        }), 404
    
    # Veriyi Series'e dönüştür
    if isinstance(data, pd.DataFrame):
        if indicator_type == 'usd_try':
            data = data['usd_try'] if 'usd_try' in data.columns else data.iloc[:, 0]
        else:
            data = data.iloc[:, 0]
    
    # MAPE FIX: Veriyi son 48 aya kısıtla (tüm göstergeler için)
    if len(data) > 48:
        logger.info(f"MAPE FIX: {indicator_type} verisi son 48 aya kısıtlanıyor ({len(data)} -> 48)")
        data = data.tail(48)
    
    # Model trainer ve predictor fonksiyonları
    def trainer(train_data):
        # Her fold için yeni forecaster oluştur
        forecaster = forecaster_class()
        
        # Validasyon sırasında test setini boş bırakmamak için minimal bir test_size kullan
        # Ancak asıl test seti zaten model_validator tarafından ayrıldığı için,
        # burada modelin içindeki train/test ayrımını minimize ediyoruz.
        # test_size=0.01 veya en az 1 örnek olacak şekilde ayarla
        ts = 0.05 if len(train_data) > 20 else 0.2
        
        try:
            forecaster.fit(train_data, test_size=ts, exogenous_data=None)
        except Exception as e:
            logger.warning(f"fit hatası ({e}), test_size=0.2 ile tekrar deneniyor")
            forecaster.fit(train_data, test_size=0.2, exogenous_data=None)
            
        return forecaster
    
    def predictor(model, train_data, steps):
        # MAPE FIX: Hibrit modeli kapalı tut (değerlendirmede kararsız)
        # Sadece Inflation ve Interest Rate forecaster'lar use_hybrid parametresini alır
        model_name = model.__class__.__name__
        if model_name in ['InflationForecaster', 'InterestRateForecaster']:
            results = model.forecast(steps=steps, use_hybrid=False)
        else:
            # ARIMAForecaster use_hybrid parametresini kabul etmez
            results = model.forecast(steps=steps)
            
        if isinstance(results, dict) and 'forecast' in results:
            return results['forecast']
        return results
    
    # Validasyon yap
    validator = ModelValidator()
    try:
        # MAPE FIX: Sadece hold-out validation kullan (cross-validation küçük veri setlerinde kararsız)
        validation_results = validator.validate_model(
            data, trainer, predictor, 'holdout', test_size, n_splits
        )
        
        # Numpy tiplerini Python native tiplerine dönüştür (JSON serialization için)
        validation_results = convert_numpy_types(validation_results)
        
        return jsonify({
            'success': True,
            'data': {
                'indicator_type': indicator_type,
                'validation_results': validation_results,
                'data_points': int(len(data))
            },
            'message': 'Model validasyonu başarıyla tamamlandı',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Validasyon hatası: {e}")
        return jsonify({
            'success': False,
            'message': f'Validasyon başarısız: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/news/analysis', methods=['GET'])
@handle_errors
def news_analysis():
    """Haber analizi yapar."""
    logger.info("Haber analizi yapılıyor...")
    
    days_back = request.args.get('days_back', 7, type=int)
    
    news_service = NewsAPIService()
    claude_service = ClaudeAPIService()
    
    # Haberleri çek
    articles = news_service.fetch_economic_news(days_back=days_back)
    
    if not articles:
        return jsonify({
            'success': False,
            'message': 'Haber çekilemedi',
            'timestamp': datetime.now().isoformat()
        }), 404
    
    # Claude ile analiz
    news_text = news_service.format_news_for_claude(articles)
    analysis = claude_service.analyze_news_sentiment(news_text)
    
    # Haberleri formatla
    formatted_articles = []
    for article in articles[:10]:
        formatted_articles.append({
            'title': article.get('title', ''),
            'description': article.get('description', ''),
            'source': article.get('source', {}).get('name', ''),
            'published_at': article.get('publishedAt', ''),
            'url': article.get('url', '')
        })
    
    return jsonify({
        'success': True,
        'data': {
            'articles': formatted_articles,
            'analysis': analysis,
            'count': len(articles)
        },
        'message': 'Haber analizi tamamlandı',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/autonomous/run', methods=['POST'])
@handle_errors
def run_autonomous_system():
    """Tam otonom tahmin sistemini çalıştırır."""
    logger.info("Otonom sistem çalıştırılıyor...")
    
    system = AutonomousForecastingSystem()
    results = system.run_full_system()
    
    if not results.get('success', False):
        return jsonify({
            'success': False,
            'message': 'Otonom sistem başarısız oldu',
            'data': results,
            'timestamp': datetime.now().isoformat()
        }), 500
    
    # Sonuçları formatla
    response_data = {
        'arima_forecast': {},
        'news_analysis': {},
        'enhanced_forecast': {},
        'metrics': {}
    }
    
    # ARIMA sonuçları
    if results.get('arima_model', {}).get('status') == 'success':
        arima_data = results['arima_model']
        response_data['arima_forecast'] = arima_data.get('forecast', {})
        response_data['metrics'] = arima_data.get('model_params', {})
    
    # News analizi
    if results.get('news_analysis', {}).get('status') == 'success':
        response_data['news_analysis'] = results['news_analysis']
    
    # Çarpanlı tahminler
    if results.get('final_forecast', {}).get('status') == 'success':
        final_data = results['final_forecast']
        response_data['enhanced_forecast'] = final_data.get('enhanced_forecast', {})
        response_data['multiplier'] = final_data.get('news_multiplier', 1.0)
    
    return jsonify({
        'success': True,
        'data': response_data,
        'message': 'Otonom sistem başarıyla tamamlandı',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/inflation/forecast', methods=['POST'])
@handle_errors
def inflation_forecast():
    """Enflasyon tahmini yapar (ARIMA + News + Claude)."""
    logger.info("Enflasyon tahmini yapılıyor (News sentiment destekli)...")
    
    params = request.get_json() or {}
    steps = params.get('steps', 12)
    test_size = params.get('test_size', 0.2)
    
    # Veri yükle
    data = load_inflation_data_from_mongodb()
    
    if data is None or data.empty:
        return jsonify({
            'success': False,
            'message': 'Enflasyon verisi bulunamadı',
            'timestamp': datetime.now().isoformat()
        }), 404
    
    # Model
    forecaster = InflationForecaster()
    forecaster.fit(data, test_size=test_size)
    
    try:
        metrics = forecaster.evaluate()
    except Exception as e:
        logger.warning(f"Metrik hesaplama hatası: {e}")
        metrics = None
    
    # ARIMA + News + Claude ile nihai tahmin
    results = forecaster.forecast_with_news(steps=steps)
    
    if results is None:
        return jsonify({
            'success': False,
            'message': 'Tahmin yapılamadı',
            'timestamp': datetime.now().isoformat()
        }), 500
    
    # Tarihsel veri
    historical = []
    for date, value in data.items():
        historical.append({
            'date': date.strftime('%Y-%m-%d'),
            'value': float(value)
        })
    
    # ARIMA tahmini
    arima_forecast = []
    for date, value in results['arima_forecast'].items():
        arima_forecast.append({
            'date': date.strftime('%Y-%m-%d'),
            'value': float(value)
        })
    
    # News çarpanlı nihai tahmin
    enhanced_forecast = []
    for date, value in results['enhanced_forecast'].items():
        enhanced_forecast.append({
            'date': date.strftime('%Y-%m-%d'),
            'value': float(value)
        })
    
    return jsonify({
        'success': True,
        'data': {
            'historical': historical,
            'arima_forecast': arima_forecast,
            'enhanced_forecast': enhanced_forecast,
            'forecast': enhanced_forecast,  # Frontend uyumluluğu için
            'multiplier': results['multiplier'],
            'news_analysis': results['news_analysis'],
            'confidence': results['confidence'],
            'metrics': metrics,
            'model_params': forecaster.best_params
        },
        'message': f"Enflasyon tahmini tamamlandı (News çarpanı: {results['multiplier']:.3f})",
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/interest_rate/forecast', methods=['POST'])
@handle_errors
def interest_rate_forecast():
    """Faiz oranı tahmini yapar (ARIMA + News + Claude)."""
    logger.info("Faiz oranı tahmini yapılıyor (News sentiment destekli)...")
    
    params = request.get_json() or {}
    steps = params.get('steps', 12)
    test_size = params.get('test_size', 0.2)
    
    # Veri yükle
    data = load_interest_rate_data_from_mongodb()
    
    if data is None or data.empty:
        return jsonify({
            'success': False,
            'message': 'Faiz oranı verisi bulunamadı',
            'timestamp': datetime.now().isoformat()
        }), 404
    
    # Model
    forecaster = InterestRateForecaster()
    forecaster.fit(data, test_size=test_size)
    
    try:
        metrics = forecaster.evaluate()
    except Exception as e:
        logger.warning(f"Metrik hesaplama hatası: {e}")
        metrics = None
    
    # ARIMA + News + Claude ile nihai tahmin
    results = forecaster.forecast_with_news(steps=steps)
    
    if results is None:
        return jsonify({
            'success': False,
            'message': 'Tahmin yapılamadı',
            'timestamp': datetime.now().isoformat()
        }), 500
    
    # Tarihsel veri
    historical = []
    for date, value in data.items():
        historical.append({
            'date': date.strftime('%Y-%m-%d'),
            'value': float(value)
        })
    
    # ARIMA tahmini
    arima_forecast = []
    for date, value in results['arima_forecast'].items():
        arima_forecast.append({
            'date': date.strftime('%Y-%m-%d'),
            'value': float(value)
        })
    
    # News çarpanlı nihai tahmin
    enhanced_forecast = []
    for date, value in results['enhanced_forecast'].items():
        enhanced_forecast.append({
            'date': date.strftime('%Y-%m-%d'),
            'value': float(value)
        })
    
    return jsonify({
        'success': True,
        'data': {
            'historical': historical,
            'arima_forecast': arima_forecast,
            'enhanced_forecast': enhanced_forecast,
            'forecast': enhanced_forecast,  # Frontend uyumluluğu için
            'multiplier': results['multiplier'],
            'news_analysis': results['news_analysis'],
            'confidence': results['confidence'],
            'metrics': metrics,
            'model_params': forecaster.best_params
        },
        'message': f"Faiz oranı tahmini tamamlandı (News çarpanı: {results['multiplier']:.3f})",
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/multi/comprehensive', methods=['GET'])
@handle_errors
def multi_comprehensive():
    """Tüm göstergeler için kapsamlı analiz."""
    import threading
    import time
    
    # G-41-G-42: Sistem yanıt süresi metrikleri
    request_start_time = time.time()
    
    logger.info("Çok göstergeli kapsamlı analiz yapılıyor...")
    
    try:
        analyzer = EconomicAnalyzer()
        
        # Timeout koruması için thread kullan
        results = {}
        error_occurred = threading.Event()
        
        def run_analysis():
            try:
                nonlocal results
                results = analyzer.analyze_all(forecast_steps=12, include_news=True)
            except Exception as e:
                logger.error(f"Analiz sırasında hata: {e}")
                logger.error(traceback.format_exc())
                error_occurred.set()
                results = {
                    'success': False,
                    'message': f'Analiz hatası: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }
        
        # Socket timeout'u artır
        import socket
        socket.setdefaulttimeout(600)
        
        analysis_thread = threading.Thread(target=run_analysis, daemon=False)  # daemon=False - thread bitene kadar bekle
        analysis_thread.start()
        analysis_thread.join(timeout=600)  # 10 dakika timeout
        
        if analysis_thread.is_alive():
            logger.error("Analiz timeout'a uğradı (10 dakika)")
            return jsonify({
                'success': False,
                'message': 'Analiz çok uzun sürdü ve timeout oldu. Lütfen daha sonra tekrar deneyin.',
                'timestamp': datetime.now().isoformat()
            }), 504
        
        if error_occurred.is_set():
            return jsonify({
                'success': False,
                'message': 'Analiz sırasında hata oluştu',
                'data': results,
                'timestamp': datetime.now().isoformat()
            }), 500
        
        if not results.get('success', False):
            return jsonify({
                'success': False,
                'message': 'Analiz başarısız oldu',
                'data': results,
                'timestamp': datetime.now().isoformat()
            }), 500
        
        # Analiz sonuçlarını MongoDB'ye kaydet
        try:
            logger.info("💾 Analiz sonuçları MongoDB'ye kaydediliyor...")
            db_manager = MongoDBManager()
            
            # MongoDB bağlantısını kontrol et
            if db_manager.database is None:
                logger.error("❌ MongoDB bağlantısı yok, kayıt yapılamıyor")
            else:
                # MongoDB'ye kaydetmeden önce numpy tiplerini dönüştür
                # Deep copy yap ve dönüştür
                import copy
                results_converted = copy.deepcopy(results)
                results_converted = convert_numpy_types(results_converted)
                # Tekrar dönüştür (nested yapılar için)
                results_converted = convert_numpy_types(results_converted)
                
                analysis_doc = {
                    'type': 'comprehensive_analysis',
                    'data': results_converted,
                    'created_at': datetime.now(),
                    'updated_at': datetime.now(),
                    'version': 1
                }
                # analysis_doc'u da dönüştür
                analysis_doc = convert_numpy_types(analysis_doc)
                collection = db_manager.get_collection('analysis_results')
                if collection is None:
                    logger.error("❌ 'analysis_results' koleksiyonu alınamadı")
                else:
                    # Eski kaydı güncelle veya yeni ekle
                    result = collection.replace_one(
                        {'type': 'comprehensive_analysis'},
                        analysis_doc,
                        upsert=True
                    )
                    logger.info(f"✅ Analiz sonuçları MongoDB'ye kaydedildi (matched: {result.matched_count}, modified: {result.modified_count}, upserted_id: {result.upserted_id})")
                    db_manager.close_connection()
        except Exception as save_error:
            logger.error(f"❌ Analiz sonuçları kaydedilemedi: {save_error}")
            logger.error(traceback.format_exc())
            # Kaydetme hatası analizi etkilemesin
        
        # G-41-G-42: Sistem yanıt süresi metrikleri
        response_time = time.time() - request_start_time
        
        # Önce numpy tiplerini dönüştür, sonra performance ekle
        results = convert_numpy_types(results)
        
        # Performance metriklerini ekle (numpy tipleri olmadan)
        results['performance'] = results.get('performance', {})
        results['performance']['api_response_time'] = float(response_time)
        results['performance']['meets_g41_target'] = bool(response_time < 3.0)  # G-41: < 3 saniye
        results['performance']['meets_g42_target'] = bool(response_time < 2.0)  # G-42: < 2 saniye
        
        # Tekrar dönüştür (performance ekledikten sonra)
        results = convert_numpy_types(results)
        
        # Response header'larına keep-alive ekle
        response = jsonify({
            'success': True,
            'data': results,
            'message': 'Çok göstergeli analiz tamamlandı',
            'timestamp': datetime.now().isoformat(),
            'response_time': float(response_time)  # G-41-G-42 metrikleri
        })
        response.headers['Connection'] = 'keep-alive'
        response.headers['Keep-Alive'] = 'timeout=600'
        response.headers['X-Response-Time'] = str(response_time)  # G-41-G-42 header
        return response
        
    except Exception as e:
        logger.error(f"multi_comprehensive endpoint hatası: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'Beklenmeyen hata: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/multi/comprehensive/latest', methods=['GET'])
@handle_errors
def get_latest_comprehensive():
    """Kaydedilmiş en son analiz sonuçlarını getirir."""
    logger.info("🔍 Kaydedilmiş analiz sonuçları getiriliyor...")
    
    try:
        db_manager = MongoDBManager()
        
        # MongoDB bağlantısını kontrol et
        if db_manager.database is None:
            logger.error("❌ MongoDB bağlantısı yok")
            return jsonify({
                'success': False,
                'message': 'MongoDB bağlantısı yok',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        collection = db_manager.get_collection('analysis_results')
        if collection is None:
            logger.error("❌ 'analysis_results' koleksiyonu alınamadı")
            return jsonify({
                'success': False,
                'message': 'Koleksiyon bulunamadı',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        logger.info("🔍 MongoDB'de analiz sonuçları aranıyor...")
        # En son kaydı getir
        latest = collection.find_one(
            {'type': 'comprehensive_analysis'},
            sort=[('updated_at', -1)]
        )
        
        if not latest:
            logger.warning("⚠️ Kaydedilmiş analiz bulunamadı")
            db_manager.close_connection()
            return jsonify({
                'success': False,
                'message': 'Kaydedilmiş analiz bulunamadı',
                'timestamp': datetime.now().isoformat()
            }), 404
        
        logger.info(f"✅ Kaydedilmiş analiz bulundu (updated_at: {latest.get('updated_at')})")
        db_manager.close_connection()
        
        # MongoDB ObjectId ve datetime'ları JSON'a uygun formata çevir
        if '_id' in latest:
            latest['_id'] = str(latest['_id'])
        
        # Datetime objelerini ISO format string'e çevir
        created_at = None
        updated_at = None
        if latest.get('created_at'):
            if isinstance(latest['created_at'], datetime):
                created_at = latest['created_at'].isoformat()
            else:
                created_at = str(latest['created_at'])
        if latest.get('updated_at'):
            if isinstance(latest['updated_at'], datetime):
                updated_at = latest['updated_at'].isoformat()
            else:
                updated_at = str(latest['updated_at'])
        
        # Numpy tiplerini Python native tiplerine dönüştür (JSON serialization için)
        data = convert_numpy_types(latest.get('data', {}))
        
        return jsonify({
            'success': True,
            'data': data,
            'created_at': created_at,
            'updated_at': updated_at,
            'message': 'Kaydedilmiş analiz sonuçları getirildi',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Kaydedilmiş analiz getirme hatası: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'Veritabanı hatası: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/visualization/comprehensive', methods=['GET'])
@handle_errors
def comprehensive_visualization():
    """Kapsamlı görselleştirme verisi döndürür."""
    logger.info("Kapsamlı görselleştirme verisi hazırlanıyor...")
    
    # Veri yükle - USD/TRY için özel yükleme
    from src.models.arima_model import load_data_from_mongodb
    data = load_data_from_mongodb(target_field='usd_try')
    
    if data is None or data.empty:
        logger.warning("Alternatif veri yükleme deneniyor...")
        data = load_complete_data_from_mongodb(target_field='usd_try')
    
    if data is None or data.empty:
        return jsonify({
            'success': False,
            'message': 'Veri bulunamadı',
            'timestamp': datetime.now().isoformat()
        }), 404
    
    # ARIMA tahmin
    forecaster = ARIMAForecaster(target_column='usd_try')
    forecaster.fit(data, test_size=0.2)
    
    # Performans değerlendirmesi (başarısız olsa bile devam et)
    try:
        metrics = forecaster.evaluate()
    except Exception as e:
        logger.warning(f"Metrik hesaplama hatası (göz ardı ediliyor): {e}")
        metrics = None
    
    forecast_results = forecaster.forecast(steps=12)
    
    # News analizi
    news_service = NewsAPIService()
    claude_service = ClaudeAPIService()
    
    try:
        articles = news_service.fetch_economic_news(days_back=7)
        news_text = news_service.format_news_for_claude(articles)
        news_analysis = claude_service.analyze_news_sentiment(news_text)
    except Exception as e:
        logger.warning(f"News analizi hatası: {e}")
        news_analysis = {'multiplier': 1.0, 'analysis': 'Analiz yapılamadı', 'confidence': 0.5}
    
    # Tarihsel veri
    historical = []
    for date, value in data.items():
        historical.append({
            'date': date.strftime('%Y-%m-%d'),
            'value': float(value)
        })
    
    # ARIMA tahminleri
    arima_forecast = []
    for date, value in forecast_results['forecast'].items():
        arima_forecast.append({
            'date': date.strftime('%Y-%m-%d'),
            'value': float(value)
        })
    
    # Çarpanlı tahminler
    multiplier = news_analysis.get('multiplier', 1.0)
    enhanced_forecast = []
    for date, value in forecast_results['forecast'].items():
        enhanced_forecast.append({
            'date': date.strftime('%Y-%m-%d'),
            'value': float(value * multiplier)
        })
    
    return jsonify({
        'success': True,
        'data': {
            'historical': historical,
            'arima_forecast': arima_forecast,
            'enhanced_forecast': enhanced_forecast,
            'metrics': metrics,
            'news_analysis': news_analysis,
            'multiplier': multiplier
        },
        'message': 'Görselleştirme verisi hazırlandı',
        'timestamp': datetime.now().isoformat()
    })

# ============================================================================
# VALIDATION ANALİZ ENDPOINT'LERİ
# ============================================================================

@app.route('/api/validation/analyze', methods=['GET'])
@handle_errors
def analyze_validation():
    """Validation sonuçlarını analiz eder ve yüksek hatalı dönemleri bulur."""
    logger.info("Validation analizi yapılıyor...")
    
    indicator_type = request.args.get('indicator_type', None)  # 'usd_try', 'inflation', 'interest_rate'
    mape_threshold = request.args.get('mape_threshold', 50.0, type=float)
    
    try:
        from src.utils.validation_analyzer import ValidationAnalyzer
        
        analyzer = ValidationAnalyzer()
        if not analyzer.connect():
            return jsonify({
                'success': False,
                'message': 'MongoDB bağlantısı kurulamadı',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        try:
            # Yüksek hatalı dönemleri bul
            if indicator_type:
                high_error = analyzer.find_high_error_periods(indicator_type, mape_threshold)
                error_df = analyzer.analyze_errors_by_date(indicator_type)
                
                return jsonify({
                    'success': True,
                    'data': {
                        'indicator_type': indicator_type,
                        'high_error_count': len(high_error),
                        'high_error_periods': high_error.to_dict('records') if not high_error.empty else [],
                        'error_history': error_df.to_dict('records') if not error_df.empty else [],
                        'mape_threshold': mape_threshold
                    },
                    'message': 'Validation analizi tamamlandı',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                # Tüm göstergeler için
                results = {}
                error_dfs = {}
                for ind in ['usd_try', 'inflation', 'interest_rate']:
                    high_error = analyzer.find_high_error_periods(ind, mape_threshold)
                    error_df = analyzer.analyze_errors_by_date(ind)
                    results[ind] = {
                        'high_error_count': len(high_error),
                        'high_error_periods': high_error.to_dict('records') if not high_error.empty else []
                    }
                    error_dfs[ind] = error_df.to_dict('records') if not error_df.empty else []
                
                return jsonify({
                    'success': True,
                    'data': {
                        'results': results,
                        'error_history': error_dfs,
                        'mape_threshold': mape_threshold
                    },
                    'message': 'Validation analizi tamamlandı',
                    'timestamp': datetime.now().isoformat()
                })
            
        finally:
            analyzer.close()
            
    except Exception as e:
        logger.error(f"Validation analiz hatası: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'Analiz hatası: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/validation/report', methods=['GET'])
@handle_errors
def get_validation_report():
    """Validation analiz raporunu döndürür."""
    logger.info("Validation raporu oluşturuluyor...")
    
    indicator_type = request.args.get('indicator_type', None)
    
    try:
        from src.utils.validation_analyzer import ValidationAnalyzer
        
        analyzer = ValidationAnalyzer()
        if not analyzer.connect():
            return jsonify({
                'success': False,
                'message': 'MongoDB bağlantısı kurulamadı',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        try:
            report = analyzer.generate_report(indicator_type=indicator_type)
            
            return jsonify({
                'success': True,
                'data': {
                    'report': report,
                    'indicator_type': indicator_type
                },
                'message': 'Validation raporu oluşturuldu',
                'timestamp': datetime.now().isoformat()
            })
            
        finally:
            analyzer.close()
            
    except Exception as e:
        logger.error(f"Validation rapor hatası: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'Rapor hatası: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/<path:path>')
def serve_frontend(path: str):
    if path.startswith('api/'):
        abort(404)

    if not _frontend_available():
        return jsonify({
            'success': False,
            'message': 'Frontend derlemesi bulunamadı. Lütfen frontend klasöründe `npm install` ve `npm run build` çalıştırın.'
        }), 503

    target = os.path.join(FRONTEND_DIST_DIR, path)
    if path and os.path.exists(target) and os.path.isfile(target):
        return send_from_directory(FRONTEND_DIST_DIR, path)
    return send_from_directory(FRONTEND_DIST_DIR, 'index.html')

# ============================================================================
# UYGULAMA BAŞLATMA
# ============================================================================

if __name__ == '__main__':
    import socket
    socket.setdefaulttimeout(600)  # 10 dakika socket timeout
    
    logger.info("=" * 60)
    logger.info("TÜBİTAK Ekonomik Göstergeler Tahmin Web Uygulaması")
    logger.info("=" * 60)
    logger.info("Sunucu başlatılıyor: http://localhost:5000")
    logger.info("=" * 60)
    
    # Flask development server için özel ayarlar
    from werkzeug.serving import WSGIRequestHandler
    
    class CustomRequestHandler(WSGIRequestHandler):
        timeout = 600  # 10 dakika timeout
        def log_request(self, code='-', size='-'):
            if hasattr(self, 'requestline'):
                logger.info(f"{self.requestline} - {code}")
    
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=5000,
        threaded=True,
        request_handler=CustomRequestHandler,
        use_reloader=False  # Reloader uzun isteklerde sorun çıkarabilir
    )

