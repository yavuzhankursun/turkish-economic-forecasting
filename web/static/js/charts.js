/**
 * Grafik Oluşturma ve Yönetimi
 * TÜBİTAK Ekonomik Göstergeler Tahmin Projesi
 */

/**
 * Ana tahmin grafiğini çizer
 */
function plotForecastChart(data, elementId = 'mainChart', indicatorType = 'usd_try') {
    const { historical, arima_forecast, enhanced_forecast, multiplier } = data;
    
    // Gösterge tipine göre birim ve başlık belirle
    const indicatorConfig = {
        'usd_try': {
            unit: 'TRY',
            title: 'USD/TRY Kuru',
            yAxisTitle: 'USD/TRY Kuru'
        },
        'inflation': {
            unit: '%',
            title: 'Enflasyon Oranı',
            yAxisTitle: 'Enflasyon Oranı (%)'
        },
        'interest_rate': {
            unit: '%',
            title: 'Faiz Oranı',
            yAxisTitle: 'Faiz Oranı (%)'
        }
    };
    
    const config = indicatorConfig[indicatorType] || indicatorConfig['usd_try'];
    
    // Tarihsel veri trace
    const historicalTrace = {
        x: historical.map(d => d.date),
        y: historical.map(d => d.value),
        type: 'scatter',
        mode: 'lines',
        name: 'Tarihsel Veri',
        line: {
            color: 'rgb(31, 119, 180)',
            width: 2
        },
        hovertemplate: `<b>Tarih:</b> %{x}<br><b>Değer:</b> %{y:.2f} ${config.unit}<extra></extra>`
    };
    
    // ARIMA tahmin trace
    const arimaTrace = {
        x: arima_forecast.map(d => d.date),
        y: arima_forecast.map(d => d.value),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'ARIMA Tahmini',
        line: {
            color: 'rgb(255, 127, 14)',
            width: 2,
            dash: 'dash'
        },
        marker: {
            size: 6
        },
        hovertemplate: `<b>Tarih:</b> %{x}<br><b>ARIMA:</b> %{y:.2f} ${config.unit}<extra></extra>`
    };
    
    // Çarpanlı tahmin trace
    const enhancedTrace = {
        x: enhanced_forecast.map(d => d.date),
        y: enhanced_forecast.map(d => d.value),
        type: 'scatter',
        mode: 'lines+markers',
        name: `News Çarpanlı (×${multiplier.toFixed(3)})`,
        line: {
            color: 'rgb(44, 160, 44)',
            width: 3
        },
        marker: {
            size: 8
        },
        hovertemplate: `<b>Tarih:</b> %{x}<br><b>Çarpanlı:</b> %{y:.2f} ${config.unit}<extra></extra>`
    };
    
    const traces = [historicalTrace, arimaTrace, enhancedTrace];
    
    const layout = {
        title: {
            text: `🎯 ${config.title} Tahmin Sistemi`,
            font: { size: 20, family: 'Arial, sans-serif' }
        },
        xaxis: {
            title: 'Tarih',
            gridcolor: 'rgba(128, 128, 128, 0.2)',
            showgrid: true
        },
        yaxis: {
            title: config.yAxisTitle,
            gridcolor: 'rgba(128, 128, 128, 0.2)',
            showgrid: true
        },
        hovermode: 'closest',
        showlegend: true,
        legend: {
            x: 0,
            y: 1,
            bgcolor: 'rgba(255, 255, 255, 0.8)',
            bordercolor: 'rgba(0, 0, 0, 0.2)',
            borderwidth: 1
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        margin: { l: 60, r: 40, t: 80, b: 60 }
    };
    
    const plotConfig = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        toImageButtonOptions: {
            format: 'png',
            filename: `forecast_${indicatorType}_${new Date().toISOString().split('T')[0]}`,
            height: 800,
            width: 1400,
            scale: 2
        }
    };
    
    Plotly.newPlot(elementId, traces, layout, plotConfig);
}

/**
 * News analizi çubuk grafiği çizer
 */
function plotNewsAnalysisChart(analysis, elementId = 'newsChart') {
    const multiplier = analysis.multiplier || 1.0;
    const confidence = analysis.confidence || 0.5;
    
    const trace = {
        x: ['News Çarpanı'],
        y: [multiplier],
        type: 'bar',
        marker: {
            color: multiplier > 1.0 ? 'rgb(44, 160, 44)' : 
                   multiplier < 1.0 ? 'rgb(214, 39, 40)' : 
                   'rgb(148, 103, 189)'
        },
        text: [multiplier.toFixed(3)],
        textposition: 'outside',
        hovertemplate: '<b>Çarpan:</b> %{y:.3f}<extra></extra>'
    };
    
    const layout = {
        title: {
            text: `📰 Haber Analizi (Güven: ${(confidence * 100).toFixed(0)}%)`,
            font: { size: 16 }
        },
        yaxis: {
            title: 'Çarpan Değeri',
            range: [0.8, 1.2]
        },
        shapes: [{
            type: 'line',
            x0: -0.5,
            x1: 0.5,
            y0: 1.0,
            y1: 1.0,
            line: {
                color: 'black',
                width: 2,
                dash: 'dash'
            }
        }],
        annotations: [{
            x: 0,
            y: 1.0,
            text: 'Nötr (1.0)',
            showarrow: false,
            yshift: 10
        }],
        plot_bgcolor: 'white',
        paper_bgcolor: 'white'
    };
    
    const config = { responsive: true };
    
    Plotly.newPlot(elementId, [trace], layout, config);
}

/**
 * Model performans metrikleri görüntüler
 */
function displayMetrics(metrics) {
    if (!metrics) {
        return '<p class="text-muted">Metrik bilgisi yok</p>';
    }
    
    let html = '<div class="row">';
    
    if (metrics.RMSE !== undefined) {
        html += `
            <div class="col-md-4 mb-3">
                <div class="metric-card">
                    <h6 class="text-muted">RMSE</h6>
                    <h3>${metrics.RMSE.toFixed(4)}</h3>
                    <small>Root Mean Squared Error</small>
                </div>
            </div>
        `;
    }
    
    if (metrics.MAE !== undefined) {
        html += `
            <div class="col-md-4 mb-3">
                <div class="metric-card">
                    <h6 class="text-muted">MAE</h6>
                    <h3>${metrics.MAE.toFixed(4)}</h3>
                    <small>Mean Absolute Error</small>
                </div>
            </div>
        `;
    }
    
    if (metrics.MSE !== undefined) {
        html += `
            <div class="col-md-4 mb-3">
                <div class="metric-card">
                    <h6 class="text-muted">MSE</h6>
                    <h3>${metrics.MSE.toFixed(4)}</h3>
                    <small>Mean Squared Error</small>
                </div>
            </div>
        `;
    }
    
    html += '</div>';
    return html;
}

/**
 * Grafiği PNG olarak indirir
 */
function downloadChart(elementId = 'mainChart') {
    Plotly.downloadImage(elementId, {
        format: 'png',
        width: 1400,
        height: 800,
        filename: `forecast_${new Date().toISOString().split('T')[0]}`
    });
}

