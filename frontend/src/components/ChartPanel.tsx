import React, { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, LineData, Time } from 'lightweight-charts';
import type { ChartSeriesPoint } from '../types';
import { formatDate } from '../utils/format';
import { useTheme } from '../contexts/ThemeContext';

interface ChartPanelProps {
  historical?: ChartSeriesPoint[];
  primary: ChartSeriesPoint[];
  secondary?: ChartSeriesPoint[];
  height?: number;
}

export function ChartPanel({ historical, primary, secondary, height = 300 }: ChartPanelProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const { theme } = useTheme();
  const [chartHeight, setChartHeight] = useState(height);

  // Mobil için responsive height
  useEffect(() => {
    const updateHeight = () => {
      if (window.innerWidth < 640) {
        setChartHeight(250);
      } else {
        setChartHeight(height);
      }
    };
    updateHeight();
    window.addEventListener('resize', updateHeight);
    return () => window.removeEventListener('resize', updateHeight);
  }, [height]);

  useEffect(() => {
    if (!containerRef.current) return;

    // Tema göre renkler
    const isDark = theme === 'dark';
    const textColor = isDark ? 'rgba(226,232,240,0.9)' : 'rgba(30,41,59,0.9)';
    const gridColor = isDark ? 'rgba(148,163,184,0.15)' : 'rgba(148,163,184,0.2)';
    const gridHColor = isDark ? 'rgba(148,163,184,0.1)' : 'rgba(148,163,184,0.15)';
    const crosshairColor = isDark ? 'rgba(148,163,184,0.3)' : 'rgba(148,163,184,0.4)';
    
    // Seri renkleri
    const historicalColor = isDark ? '#94a3b8' : '#64748b';
    const forecastLineColor = isDark ? '#6366F1' : '#2563eb';
    const forecastTopColor = isDark ? 'rgba(99, 102, 241, 0.45)' : 'rgba(37, 99, 235, 0.35)';
    const forecastBottomColor = isDark ? 'rgba(99, 102, 241, 0.05)' : 'rgba(37, 99, 235, 0.05)';
    const enhancedColor = isDark ? '#22d3ee' : '#0891b2';

    // Container genişliğini al
    const getContainerWidth = () => {
      if (containerRef.current) {
        return containerRef.current.clientWidth || containerRef.current.offsetWidth || 400;
      }
      return 400;
    };

    // İlk genişliği al
    const containerWidth = getContainerWidth();

    const chart = createChart(containerRef.current, {
      layout: {
        background: { color: 'transparent' },
        textColor,
        attributionLogo: false
      },
      width: containerWidth,
      height: chartHeight,
      autoSize: false,
      grid: {
        vertLines: { color: gridColor },
        horzLines: { color: gridHColor }
      },
      rightPriceScale: {
        borderVisible: false,
        scaleMargins: {
          top: 0.1,
          bottom: 0.1
        }
      },
      leftPriceScale: {
        visible: false
      },
      timeScale: {
        borderVisible: false,
        timeVisible: true,
        rightOffset: 4,
        barSpacing: 4,
        minBarSpacing: 2,
        tickMarkFormatter: (time: Time) => formatDate(time as string)
      },
      crosshair: {
        mode: 0,
        vertLine: {
          width: 1,
          color: crosshairColor,
          style: 1
        },
        horzLine: {
          width: 1,
          color: crosshairColor,
          style: 1
        }
      },
      watermark: {
        visible: false
      }
    });

    chartRef.current = chart;

    const historicalSeries = historical?.length
      ? chart.addLineSeries({
          color: historicalColor,
          lineWidth: 2,
          lineStyle: 0,
          priceScaleId: 'right'
        })
      : null;

    const forecastSeries = chart.addAreaSeries({
      lineColor: forecastLineColor,
      topColor: forecastTopColor,
      bottomColor: forecastBottomColor,
      lineWidth: 2,
      priceScaleId: 'right'
    });

    const enhancedSeries = secondary
      ? chart.addLineSeries({
          color: enhancedColor,
          lineWidth: 2,
          lineStyle: 2,
          priceScaleId: 'right'
        })
      : null;

    if (historicalSeries && historical) {
      historicalSeries.setData(toLineData(historical));
    }

    forecastSeries.setData(toLineData(primary));
    if (enhancedSeries && secondary) {
      enhancedSeries.setData(toLineData(secondary));
    }

    chart.timeScale().fitContent();

    // TradingView logo/attribution'ı DOM'dan kaldır
    const removeTradingViewLogos = () => {
      if (containerRef.current) {
        const links = containerRef.current.querySelectorAll('a[href*="tradingview.com"], a[title*="TradingView"]');
        links.forEach(link => {
          link.remove();
        });
        // SVG içindeki logo elementlerini de kaldır
        const svgs = containerRef.current.querySelectorAll('svg');
        svgs.forEach(svg => {
          const title = svg.querySelector('title');
          if (title && title.textContent?.includes('TradingView')) {
            svg.remove();
          }
        });
      }
    };

    // Hemen kaldır ve birkaç kez daha dene (DOM gecikmesi için)
    removeTradingViewLogos();
    setTimeout(removeTradingViewLogos, 100);
    setTimeout(removeTradingViewLogos, 500);
    setTimeout(removeTradingViewLogos, 1000);

    const handleResize = () => {
      if (containerRef.current && chartRef.current) {
        const newWidth = containerRef.current.clientWidth || containerRef.current.offsetWidth;
        if (newWidth > 0) {
          chartRef.current.applyOptions({ width: newWidth });
          // Grafik genişliği değiştiğinde içeriği tekrar fit et
          requestAnimationFrame(() => {
            if (chartRef.current) {
              chartRef.current.timeScale().fitContent();
            }
          });
        }
      }
    };

    // İlk resize'ı biraz gecikmeyle yap (DOM'un tam render olmasını bekle)
    // Birden fazla deneme yaparak container'ın gerçek genişliğini al
    const initialResize = () => {
      handleResize();
      // Birkaç kez daha dene
      setTimeout(handleResize, 50);
      setTimeout(handleResize, 200);
      setTimeout(handleResize, 500);
    };
    
    requestAnimationFrame(() => {
      setTimeout(initialResize, 100);
    });
    
    // ResizeObserver kullanarak daha hassas resize takibi
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.target === containerRef.current) {
          handleResize();
          // Resize sırasında logo'ları tekrar kaldır
          removeTradingViewLogos();
        }
      }
    });

    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
      
      // MutationObserver ile DOM değişikliklerini izle ve logo'ları kaldır
      const mutationObserver = new MutationObserver(() => {
        removeTradingViewLogos();
      });
      
      mutationObserver.observe(containerRef.current, {
        childList: true,
        subtree: true,
        attributes: false
      });
      
      // Cleanup için mutationObserver'ı da sakla
      (chartRef.current as any)._mutationObserver = mutationObserver;
    }

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      resizeObserver.disconnect();
      if (chartRef.current) {
        const mutationObserver = (chartRef.current as any)?._mutationObserver;
        if (mutationObserver) {
          mutationObserver.disconnect();
        }
        chartRef.current.remove();
        chartRef.current = null;
      }
      // Son bir kez logo'ları kaldır
      removeTradingViewLogos();
    };
  }, [historical, primary, secondary, chartHeight, theme]);

  return (
    <div 
      ref={containerRef} 
      className="w-full chart-container" 
      style={{ 
        minHeight: `${chartHeight}px`,
        width: '100%',
        maxWidth: '100%',
        position: 'relative',
        flex: '1 1 auto',
        minWidth: 0
      }} 
    />
  );
}

function toLineData(points: ChartSeriesPoint[]): LineData[] {
  return points
    .filter((point) => point.value !== null && !Number.isNaN(point.value))
    .map((point) => ({ time: point.time, value: Number(point.value) }));
}

