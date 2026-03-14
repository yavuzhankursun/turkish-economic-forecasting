/**
 * Bar Chart Component
 * ===================
 * 
 * G-14 Gereksinimi: Sütun grafikleri görselleştirmesi
 * 
 * Lightweight-charts kullanarak sütun grafikleri oluşturur.
 */

import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi, ISeriesApi, BarData, Time } from 'lightweight-charts';
import { useTheme } from '../contexts/ThemeContext';

export interface BarChartProps {
  data: Array<{ time: string; value: number; label?: string }>;
  height?: number;
  title?: string;
}

export function BarChart({ data, height = 300, title }: BarChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Bar'> | null>(null);
  const { theme } = useTheme();

  useEffect(() => {
    if (!containerRef.current) return;

    const textColor = theme === 'dark' ? '#E5E7EB' : '#1F2937';
    const gridColor = theme === 'dark' ? '#374151' : '#E5E7EB';
    const barColor = theme === 'dark' ? '#3B82F6' : '#2563EB';

    // Chart oluştur (width/height verildiği için autoSize: false)
    const chart = createChart(containerRef.current, {
      layout: {
        background: { color: 'transparent' },
        textColor,
        attributionLogo: false
      },
      width: containerRef.current.clientWidth,
      height,
      autoSize: false,
      grid: {
        vertLines: { color: gridColor },
        horzLines: { color: gridColor }
      },
      rightPriceScale: {
        borderVisible: false
      },
      timeScale: {
        borderVisible: false,
        timeVisible: true
      },
      watermark: {
        visible: false
      }
    });

    chartRef.current = chart;

    // Bar serisi ekle
    const barSeries = chart.addBarSeries({
      upColor: barColor,
      downColor: barColor,
      priceFormat: {
        type: 'price',
        precision: 2,
        minMove: 0.01
      }
    });

    seriesRef.current = barSeries;

    // Veriyi dönüştür ve ekle
    const chartData: BarData[] = data.map(item => ({
      time: item.time as Time,
      open: item.value,
      high: item.value,
      low: 0,
      close: item.value
    }));

    barSeries.setData(chartData);

    // Responsive
    const handleResize = () => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: containerRef.current.clientWidth
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
      }
    };
  }, [data, height, theme]);

  return (
    <div className="w-full">
      {title && (
        <h3 className="text-lg font-semibold mb-2 text-slate-900 dark:text-slate-100">
          {title}
        </h3>
      )}
      <div ref={containerRef} className="w-full" style={{ height: `${height}px` }} />
    </div>
  );
}

