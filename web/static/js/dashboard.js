/**
 * Dashboard Ana Mantığı
 * TÜBİTAK Ekonomik Göstergeler Tahmin Projesi
 */

// Mevcut seçili gösterge
let currentIndicator = 'usd_try';

// Sayfa yüklendiğinde
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard yüklendi');
    
    // İlk yüklemeler
    checkHealth();
    loadLatestData();
    loadNewsHeadlines();
    loadCurrentIndicator();
});

/**
 * Toast bildirimi gösterir
 */
function showToast(title, message, type = 'info') {
    const toastElement = document.getElementById('liveToast');
    const toastTitle = document.getElementById('toastTitle');
    const toastBody = document.getElementById('toastBody');
    
    toastTitle.textContent = title;
    toastBody.textContent = message;
    
    // Renk değiştir
    toastElement.classList.remove('bg-success', 'bg-danger', 'bg-warning', 'bg-info');
    if (type === 'success') toastElement.classList.add('bg-success', 'text-white');
    else if (type === 'error') toastElement.classList.add('bg-danger', 'text-white');
    else if (type === 'warning') toastElement.classList.add('bg-warning');
    else toastElement.classList.add('bg-info', 'text-white');
    
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
}

/**
 * Loading modal gösterir/gizler
 */
function showLoading(show, message = 'İşlem devam ediyor...') {
    const modalEl = document.getElementById('loadingModal');
    const messageEl = document.getElementById('loadingMessage');
    
    if (show) {
        messageEl.textContent = message;
        const modal = new bootstrap.Modal(modalEl);
        modal.show();
    } else {
        // Modal'ı kapat ve backdrop'u temizle
        const modal = bootstrap.Modal.getInstance(modalEl);
        if (modal) {
            modal.hide();
        }
        // Backdrop'u manuel temizle
        const backdrop = document.querySelector('.modal-backdrop');
        if (backdrop) {
            backdrop.remove();
        }
        document.body.classList.remove('modal-open');
        document.body.style.removeProperty('overflow');
        document.body.style.removeProperty('padding-right');
    }
}

/**
 * Sistem sağlık kontrolü
 */
async function checkHealth() {
    try {
        const result = await checkSystemHealth();
        
        if (result.success) {
            updateHealthStatus(result.data.status);
            console.log('Sağlık kontrolü:', result.data.messages);
        }
    } catch (error) {
        console.error('Sağlık kontrolü hatası:', error);
        showToast('Hata', 'Sağlık kontrolü yapılamadı: ' + error.message, 'error');
    }
}

/**
 * Sağlık durumu kartlarını günceller
 */
function updateHealthStatus(status) {
    const healthDiv = document.getElementById('healthStatus');
    
    const systems = [
        { key: 'mongodb', icon: 'database', name: 'MongoDB' },
        { key: 'news_api', icon: 'newspaper', name: 'News API' },
        { key: 'claude_api', icon: 'robot', name: 'Claude API' },
        { key: 'arima_model', icon: 'graph-up', name: 'ARIMA Model' }
    ];
    
    healthDiv.innerHTML = systems.map(sys => {
        const isOk = status[sys.key];
        const badgeClass = isOk ? 'bg-success' : 'bg-danger';
        const badgeText = isOk ? 'OK' : 'Hata';
        
        return `
            <div class="col-md-3">
                <div class="health-card">
                    <i class="bi bi-${sys.icon}"></i>
                    <h6>${sys.name}</h6>
                    <span class="badge ${badgeClass}">${badgeText}</span>
                </div>
            </div>
        `;
    }).join('');
}

/**
 * En son verileri yükler
 */
async function loadLatestData() {
    try {
        const result = await getLatestData();
        
        if (result.success) {
            updateDataTable(result.data.records);
        }
    } catch (error) {
        console.error('Veri yükleme hatası:', error);
        const tbody = document.querySelector('#dataTable tbody');
        tbody.innerHTML = '<tr><td colspan="2" class="text-center text-danger">Veri yüklenemedi</td></tr>';
    }
}

/**
 * Veri tablosunu günceller
 */
function updateDataTable(records) {
    const tbody = document.querySelector('#dataTable tbody');
    
    tbody.innerHTML = records.reverse().map(record => `
        <tr>
            <td>${record.date}</td>
            <td><strong>${record.value.toFixed(2)} TRY</strong></td>
        </tr>
    `).join('');
}

/**
 * ARIMA tahmini yapar
 */
async function runARIMAForecast() {
    showLoading(true, 'ARIMA modeli eğitiliyor ve tahmin yapılıyor...');
    
    try {
        const result = await getARIMAForecast(12, 0.2);
        
        if (result.success) {
            showToast('Başarılı', result.message, 'success');
            
            // Grafik çiz
            plotForecastChart({
                historical: result.data.historical,
                arima_forecast: result.data.forecast,
                enhanced_forecast: result.data.forecast, // Sadece ARIMA
                multiplier: 1.0
            });
            
            // Metrikleri göster
            const metricsDiv = document.getElementById('metricsDisplay');
            metricsDiv.innerHTML = displayMetrics(result.data.metrics);
        }
    } catch (error) {
        console.error('ARIMA hatası:', error);
        showToast('Hata', 'ARIMA tahmini yapılamadı: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

/**
 * Haber analizi yapar
 */
async function runNewsAnalysis() {
    showLoading(true, 'Haberler çekiliyor ve analiz ediliyor...');
    
    try {
        const result = await getNewsAnalysis(7);
        
        if (result.success) {
            showToast('Başarılı', `${result.data.count} haber analiz edildi`, 'success');
            
            // Analiz sonucunu göster
            const analysis = result.data.analysis;
            const message = `
                <strong>Analiz:</strong> ${analysis.analysis}<br>
                <strong>Çarpan:</strong> ${analysis.multiplier.toFixed(3)}<br>
                <strong>Güven:</strong> ${(analysis.confidence * 100).toFixed(0)}%
            `;
            
            showToast('Haber Analizi', message, 'info');
        }
    } catch (error) {
        console.error('Haber analizi hatası:', error);
        showToast('Hata', 'Haber analizi yapılamadı: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

/**
 * Otonom sistemi çalıştırır
 */
async function runAutonomousSystem() {
    showLoading(true, 'Otonom sistem çalışıyor (2-3 dakika sürebilir)...');
    
    try {
        const result = await runAutonomous();
        
        if (result.success) {
            showToast('Başarılı', result.message, 'success');
            
            // Sonuçları göster
            const data = result.data;
            
            // Grafik çiz
            if (data.arima_forecast && data.enhanced_forecast) {
                // Veriyi uygun formata çevir
                const historical = await getLatestData();
                
                const arimaForecast = Object.entries(data.arima_forecast).map(([date, value]) => ({
                    date,
                    value
                }));
                
                const enhancedForecast = Object.entries(data.enhanced_forecast).map(([date, value]) => ({
                    date,
                    value
                }));
                
                plotForecastChart({
                    historical: historical.data.records,
                    arima_forecast: arimaForecast,
                    enhanced_forecast: enhancedForecast,
                    multiplier: data.multiplier || 1.0
                });
            }
        }
    } catch (error) {
        console.error('Otonom sistem hatası:', error);
        showToast('Hata', 'Otonom sistem çalıştırılamadı: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

/**
 * Haber başlıklarını yükler
 */
async function loadNewsHeadlines() {
    const newsContainer = document.getElementById('newsContainer');
    
    try {
        const result = await getNewsAnalysis(7);
        
        if (result.success && result.data.articles) {
            const articles = result.data.articles.slice(0, 5); // İlk 5 haber
            const analysis = result.data.analysis;
            
            // Haberleri göster
            newsContainer.innerHTML = articles.map((article, index) => `
                <div class="news-card" onclick="window.open('${article.url}', '_blank')">
                    <div class="news-card-title">${article.title}</div>
                    <div class="news-card-description">${article.description || 'Açıklama yok'}</div>
                    <div class="news-card-meta">
                        <span class="news-source">${article.source}</span>
                        <small>${new Date(article.published_at).toLocaleDateString('tr-TR')}</small>
                    </div>
                </div>
            `).join('');
            
            // Sentiment göster
            updateSentimentDisplay(analysis);
        } else {
            newsContainer.innerHTML = `
                <div class="text-center py-4">
                    <i class="bi bi-newspaper" style="font-size: 3rem; color: #ddd;"></i>
                    <p class="text-muted mt-2">Haber çekilemedi</p>
                    <p class="text-muted small">API anahtarınızı kontrol edin</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('Haber yükleme hatası:', error);
        newsContainer.innerHTML = `
            <div class="text-center py-4">
                <i class="bi bi-exclamation-triangle" style="font-size: 3rem; color: #ff9500;"></i>
                <p class="text-muted mt-2">Haber yüklenemedi</p>
                <button class="btn btn-sm btn-primary mt-2" onclick="loadNewsHeadlines()">
                    <i class="bi bi-arrow-clockwise"></i> Tekrar Dene
                </button>
            </div>
        `;
    }
}

/**
 * Sentiment analizi gösterir
 */
function updateSentimentDisplay(analysis) {
    const sentimentDiv = document.getElementById('sentimentDisplay');
    
    if (!analysis || !analysis.multiplier) {
        return;
    }
    
    const multiplier = analysis.multiplier;
    const confidence = analysis.confidence || 0.5;
    
    // Sentiment durumu
    let sentimentClass = 'sentiment-neutral';
    let sentimentText = 'Nötr';
    let sentimentIcon = 'emoji-neutral';
    
    if (multiplier > 1.0) {
        sentimentClass = 'sentiment-positive';
        sentimentText = 'Pozitif';
        sentimentIcon = 'emoji-smile';
    } else if (multiplier < 1.0) {
        sentimentClass = 'sentiment-negative';
        sentimentText = 'Negatif';
        sentimentIcon = 'emoji-frown';
    }
    
    sentimentDiv.innerHTML = `
        <div class="text-center">
            <i class="bi bi-${sentimentIcon}" style="font-size: 3rem; color: var(--primary-color);"></i>
            <h4 class="mt-3 mb-2">
                <span class="sentiment-badge ${sentimentClass}">${sentimentText}</span>
            </h4>
            <p class="mb-2"><strong>Çarpan:</strong> ${multiplier.toFixed(3)}</p>
            <div class="progress" style="height: 8px; border-radius: 10px;">
                <div class="progress-bar" role="progressbar" 
                     style="width: ${confidence * 100}%; background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));"
                     aria-valuenow="${confidence * 100}" aria-valuemin="0" aria-valuemax="100"></div>
            </div>
            <small class="text-muted">Güven: ${(confidence * 100).toFixed(0)}%</small>
            <p class="mt-3 small text-muted">${analysis.analysis || ''}</p>
        </div>
    `;
}

/**
 * Metrikleri gösterir
 */
function displayMetrics(metrics) {
    return `
        <div class="row">
            <div class="col-md-3">
                <div class="metric-item">
                    <strong>RMSE:</strong> ${(metrics.RMSE || 0).toFixed(4)}
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-item">
                    <strong>MAE:</strong> ${(metrics.MAE || 0).toFixed(4)}
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-item">
                    <strong>MSE:</strong> ${(metrics.MSE || 0).toFixed(4)}
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-item">
                    <strong>MAPE:</strong> ${(metrics.MAPE || 0).toFixed(2)}%
                </div>
            </div>
        </div>
    `;
}

/**
 * Gösterge değiştirme
 */
function changeIndicator() {
    const selector = document.getElementById('indicatorSelector');
    currentIndicator = selector.value;
    loadCurrentIndicator();
}

/**
 * Mevcut göstergeyi yükle
 */
async function loadCurrentIndicator() {
    const indicatorNames = {
        'usd_try': 'USD/TRY',
        'inflation': 'Enflasyon',
        'interest_rate': 'Faiz Oranı'
    };
    
    showLoading(true, `${indicatorNames[currentIndicator]} tahmini yükleniyor...`);
    
    try {
        let result;
        
        if (currentIndicator === 'usd_try') {
            result = await getComprehensiveVisualization();
        } else if (currentIndicator === 'inflation') {
            const response = await fetch('/api/inflation/forecast', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ steps: 12, test_size: 0.2 })
            });
            result = await response.json();
        } else if (currentIndicator === 'interest_rate') {
            const response = await fetch('/api/interest_rate/forecast', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ steps: 12, test_size: 0.2 })
            });
            result = await response.json();
        }
        
        if (result.success) {
            // Veriyi formatla
            let chartData;
            
            if (currentIndicator === 'usd_try') {
                chartData = result.data;
            } else {
                // Enflasyon ve Faiz için format dönüşümü - ARIMA ve Enhanced ayrı
                chartData = {
                    historical: result.data.historical,
                    arima_forecast: result.data.arima_forecast || result.data.forecast,
                    enhanced_forecast: result.data.enhanced_forecast || result.data.forecast,
                    multiplier: result.data.multiplier || 1.0
                };
            }
            
            plotForecastChart(chartData, 'mainChart', currentIndicator);
            
            // Metrikleri göster
            const metricsDiv = document.getElementById('metricsDisplay');
            if (result.data.metrics) {
                metricsDiv.innerHTML = displayMetrics(result.data.metrics);
            }
            
            showToast('Başarılı', `${indicatorNames[currentIndicator]} yüklendi`, 'success');
        }
    } catch (error) {
        console.error('Gösterge yükleme hatası:', error);
        showToast('Hata', 'Gösterge yüklenemedi: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

/**
 * Kapsamlı görselleştirme yükler
 */
async function loadComprehensiveVisualization() {
    showLoading(true, 'Kapsamlı görselleştirme hazırlanıyor...');
    
    try {
        const result = await getComprehensiveVisualization();
        
        if (result.success) {
            const data = result.data;
            
            console.log('Görselleştirme verisi:', data);
            
            // Ana grafiği çiz
            try {
                plotForecastChart(data);
                console.log('Grafik başarıyla çizildi');
            } catch (chartError) {
                console.error('Grafik çizim hatası:', chartError);
                throw new Error('Grafik çizilemedi: ' + chartError.message);
            }
            
            // Metrikleri göster
            const metricsDiv = document.getElementById('metricsDisplay');
            if (data.metrics) {
                metricsDiv.innerHTML = displayMetrics(data.metrics);
            } else {
                metricsDiv.innerHTML = '<p class="text-muted">Model metrikleri hesaplanamadı</p>';
            }
            
            // Sentiment göster
            if (data.news_analysis) {
                updateSentimentDisplay(data.news_analysis);
            }
            
            showToast('Başarılı', 'Görselleştirme yüklendi', 'success');
        } else {
            throw new Error(result.message || 'Görselleştirme başarısız');
        }
    } catch (error) {
        console.error('Görselleştirme hatası:', error);
        showToast('Uyarı', 'Görselleştirme yüklenemedi: ' + error.message, 'warning');
        
        // Boş grafik göster
        document.getElementById('mainChart').innerHTML = `
            <div class="alert alert-info text-center">
                <i class="bi bi-info-circle"></i> 
                <p><strong>Görselleştirme yüklenemedi</strong></p>
                <p>Lütfen önce veri olduğundan emin olun veya ARIMA tahminini çalıştırın.</p>
                <p class="text-muted small">Hata: ${error.message}</p>
            </div>
        `;
    } finally {
        showLoading(false);
    }
}

