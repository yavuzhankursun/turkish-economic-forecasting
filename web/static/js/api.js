/**
 * API Yardımcı Fonksiyonları
 * TÜBİTAK Ekonomik Göstergeler Tahmin Projesi
 */

const API_BASE = '';
const API_TIMEOUT = 60000; // 60 saniye

/**
 * API çağrısı yapar (timeout ve retry desteği ile)
 */
async function apiCall(endpoint, options = {}) {
    const { method = 'GET', body = null, timeout = API_TIMEOUT, retries = 2 } = options;
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    const config = {
        method,
        headers: {
            'Content-Type': 'application/json',
        },
        signal: controller.signal
    };
    
    if (body) {
        config.body = JSON.stringify(body);
    }
    
    let lastError;
    
    for (let attempt = 0; attempt <= retries; attempt++) {
        try {
            const response = await fetch(`${API_BASE}${endpoint}`, config);
            clearTimeout(timeoutId);
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.message || `HTTP ${response.status}`);
            }
            
            return data;
        } catch (error) {
            lastError = error;
            
            if (error.name === 'AbortError') {
                console.error('İstek zaman aşımına uğradı');
                throw new Error('İstek zaman aşımına uğradı. Lütfen tekrar deneyin.');
            }
            
            if (attempt < retries) {
                console.log(`Tekrar deneniyor... (${attempt + 1}/${retries})`);
                await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
            }
        }
    }
    
    clearTimeout(timeoutId);
    throw lastError;
}

/**
 * Sağlık kontrolü yapar
 */
async function checkSystemHealth() {
    return await apiCall('/api/health');
}

/**
 * En son verileri getirir
 */
async function getLatestData() {
    return await apiCall('/api/data/latest');
}

/**
 * ARIMA tahmini yapar
 */
async function getARIMAForecast(steps = 12, testSize = 0.2) {
    return await apiCall('/api/arima/forecast', {
        method: 'POST',
        body: { steps, test_size: testSize }
    });
}

/**
 * Haber analizi yapar
 */
async function getNewsAnalysis(daysBack = 7) {
    return await apiCall(`/api/news/analysis?days_back=${daysBack}`);
}

/**
 * Otonom sistemi çalıştırır
 */
async function runAutonomous() {
    return await apiCall('/api/autonomous/run', {
        method: 'POST',
        timeout: 120000 // 2 dakika
    });
}

/**
 * Kapsamlı görselleştirme verisi getirir
 */
async function getComprehensiveVisualization() {
    return await apiCall('/api/visualization/comprehensive', {
        timeout: 90000 // 90 saniye
    });
}

