/**
 * Heatmap Chart Component
 * =======================
 * 
 * G-14 Gereksinimi: Isı haritası görselleştirmesi
 * 
 * Korelasyon matrisi ve benzer veriler için ısı haritası oluşturur.
 */

import React, { useMemo } from 'react';
import { useTheme } from '../contexts/ThemeContext';

export interface HeatmapChartProps {
  data: Array<{ x: string; y: string; value: number }>;
  xLabels: string[];
  yLabels: string[];
  title?: string;
}

export function HeatmapChart({ data, xLabels, yLabels, title }: HeatmapChartProps) {
  const { theme } = useTheme();

  // Veriyi matris formatına dönüştür
  const matrix = useMemo(() => {
    const mat: number[][] = yLabels.map(() => new Array(xLabels.length).fill(0));
    
    data.forEach(item => {
      const xIdx = xLabels.indexOf(item.x);
      const yIdx = yLabels.indexOf(item.y);
      if (xIdx >= 0 && yIdx >= 0) {
        mat[yIdx][xIdx] = item.value;
      }
    });
    
    return mat;
  }, [data, xLabels, yLabels]);

  // Renk hesaplama
  const getColor = (value: number, min: number, max: number) => {
    if (max === min) return theme === 'dark' ? '#3B82F6' : '#2563EB';
    
    const normalized = (value - min) / (max - min);
    
    if (theme === 'dark') {
      // Dark theme: mavi tonları
      const intensity = Math.floor(normalized * 255);
      return `rgb(${intensity}, ${intensity + 50}, 255)`;
    } else {
      // Light theme: mavi tonları
      const intensity = Math.floor(normalized * 200);
      return `rgb(${intensity}, ${intensity + 55}, 255)`;
    }
  };

  const minValue = Math.min(...data.map(d => d.value));
  const maxValue = Math.max(...data.map(d => d.value));

  return (
    <div className="w-full">
      {title && (
        <h3 className="text-lg font-semibold mb-4 text-slate-900 dark:text-slate-100">
          {title}
        </h3>
      )}
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <th className="p-2 border border-slate-300 dark:border-slate-600 bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-100"></th>
              {xLabels.map(label => (
                <th
                  key={label}
                  className="p-2 border border-slate-300 dark:border-slate-600 bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-100 text-sm"
                >
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {yLabels.map((yLabel, yIdx) => (
              <tr key={yLabel}>
                <td className="p-2 border border-slate-300 dark:border-slate-600 bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-100 text-sm font-medium">
                  {yLabel}
                </td>
                {xLabels.map((xLabel, xIdx) => {
                  const value = matrix[yIdx][xIdx];
                  const bgColor = getColor(value, minValue, maxValue);
                  return (
                    <td
                      key={`${xLabel}-${yLabel}`}
                      className="p-3 border border-slate-300 dark:border-slate-600 text-center text-slate-900 dark:text-slate-100 text-sm"
                      style={{ backgroundColor: bgColor }}
                    >
                      {value.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-4 flex items-center justify-center gap-4">
        <div className="flex items-center gap-2">
          <div
            className="w-4 h-4 rounded"
            style={{ backgroundColor: getColor(minValue, minValue, maxValue) }}
          />
          <span className="text-sm text-slate-600 dark:text-slate-400">
            {minValue.toFixed(2)}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div
            className="w-4 h-4 rounded"
            style={{ backgroundColor: getColor(maxValue, minValue, maxValue) }}
          />
          <span className="text-sm text-slate-600 dark:text-slate-400">
            {maxValue.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
}

