import React, { ReactNode, useMemo } from 'react';
import { formatNumber, formatPercent } from '../utils/format';
import type { IndicatorForecast } from '../types';
import type { ValidationResponse } from '../api';

interface IndicatorCardProps {
  title: string;
  currency?: boolean;
  indicator: IndicatorForecast | null;
  chart: ReactNode;
  newsMultiplier?: number;
  newsSummary?: string;
  validationData?: ValidationResponse | null;
}

export function IndicatorCard({
  title,
  currency = false,
  indicator,
  chart,
  newsMultiplier,
  newsSummary,
  validationData
}: IndicatorCardProps) {
  const metrics = indicator?.metrics;

  const formattedMultiplier = useMemo(() => {
    if (newsMultiplier === undefined || newsMultiplier === null) {
      return null;
    }
    return newsMultiplier.toFixed(3);
  }, [newsMultiplier]);

  return (
    <section className="print-report-card rounded-2xl sm:rounded-3xl bg-white/90 dark:bg-slate-900/80 backdrop-blur-xl border border-blue-200 dark:border-slate-800 shadow-panel overflow-hidden flex flex-col w-full">
      <header className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 sm:gap-2 px-3 sm:px-4 xl:px-6 pt-3 sm:pt-4 xl:pt-6">
        <div className="min-w-0 flex-1">
          <h2 className="text-base sm:text-lg font-semibold tracking-tight text-slate-900 dark:text-white">{title}</h2>
          <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400">ARIMA + Sonnet 4.5 haber çarpanı</p>
        </div>
        <div className="text-left sm:text-right flex-shrink-0">
          <p className="text-[10px] sm:text-xs uppercase tracking-widest text-slate-500">Son Değer</p>
          <p className="text-xl sm:text-2xl font-bold text-blue-600 dark:text-brand-light">
            {formatNumber(indicator?.last_value ?? null, currency ? { style: 'currency', currency: 'TRY' } : {})}
          </p>
        </div>
      </header>

      <div className="px-3 sm:px-4 xl:px-6 pt-3 sm:pt-4 pb-4 sm:pb-6 flex flex-col lg:flex-row gap-3 sm:gap-4 xl:gap-6">
        <div className="flex-1 min-h-[250px] sm:min-h-[300px] w-full overflow-visible" style={{ minWidth: 0, flexBasis: 'auto', flexGrow: 1 }}>{chart}</div>

        <aside className="w-full lg:w-64 flex-shrink-0 flex flex-col gap-3 sm:gap-4">
          <div className="rounded-xl sm:rounded-2xl border border-blue-200 dark:border-slate-800 bg-blue-50/50 dark:bg-slate-900/60 p-2.5 sm:p-3">
            <p className="text-[10px] sm:text-xs uppercase tracking-wide text-slate-600 dark:text-slate-500 mb-1.5 sm:mb-2">Model metrikleri</p>
            <dl className="grid grid-cols-2 gap-1.5 sm:gap-2 text-[10px] sm:text-xs text-slate-700 dark:text-slate-300">
              <MetricItem label="RMSE" value={formatNumber(metrics?.RMSE ?? null)} />
              <MetricItem label="MAPE" value={formatPercent(metrics?.MAPE ?? null)} />
              <MetricItem label="MAE" value={formatNumber(metrics?.MAE ?? null)} />
              <MetricItem label="Örnek" value={metrics?.samples?.toString() ?? '—'} />
            </dl>
          </div>

          <div className="rounded-xl sm:rounded-2xl border border-blue-200 dark:border-slate-800 bg-blue-50/50 dark:bg-slate-900/60 p-2.5 sm:p-3">
            <p className="text-[10px] sm:text-xs uppercase tracking-wide text-slate-600 dark:text-slate-500 mb-1">Sonnet 4.5 haber çarpanı</p>
            <p className="text-sm sm:text-base font-semibold text-blue-600 dark:text-brand-light">{formattedMultiplier ?? '—'}</p>
            <p className="text-[9px] sm:text-[10px] text-slate-600 dark:text-slate-400 mt-1 leading-relaxed line-clamp-3">
              {newsSummary ?? 'Haber analizi henüz yüklenmedi.'}
            </p>
          </div>

          {validationData && !validationData.success && validationData.message && (
            <div className="rounded-xl sm:rounded-2xl border border-amber-200 dark:border-amber-800 bg-amber-50/50 dark:bg-amber-900/20 p-2.5 sm:p-3">
              <p className="text-[10px] sm:text-xs uppercase tracking-wide text-slate-600 dark:text-slate-500 mb-1">Model Validasyonu</p>
              <p className="text-[10px] sm:text-xs text-amber-700 dark:text-amber-400">{validationData.message}</p>
            </div>
          )}
          {validationData?.success && validationData.data?.validation_results && (
            <div className="rounded-xl sm:rounded-2xl border border-green-200 dark:border-green-800 bg-green-50/50 dark:bg-green-900/60 p-2.5 sm:p-3">
              <p className="text-[10px] sm:text-xs uppercase tracking-wide text-slate-600 dark:text-slate-500 mb-1.5 sm:mb-2">Model Validasyonu</p>
              {validationData.data.validation_results.hold_out && !validationData.data.validation_results.hold_out.error && (
                <div className="mb-1.5 sm:mb-2">
                  <p className="text-[9px] sm:text-[10px] uppercase text-slate-500 dark:text-slate-400 mb-1">Hold-Out</p>
                  <dl className="grid grid-cols-2 gap-1 text-[9px] sm:text-[10px] text-slate-700 dark:text-slate-300">
                    <div>
                      <span className="text-slate-500">RMSE:</span>
                      <span className="ml-1 font-semibold">{formatNumber(validationData.data.validation_results.hold_out.RMSE ?? null)}</span>
                    </div>
                    <div>
                      <span className="text-slate-500">MAPE:</span>
                      <span className="ml-1 font-semibold">{formatPercent(validationData.data.validation_results.hold_out.MAPE ?? null)}</span>
                    </div>
                  </dl>
                </div>
              )}
              {validationData.data.validation_results.cross_validation && !validationData.data.validation_results.cross_validation.error && (
                <div>
                  <p className="text-[9px] sm:text-[10px] uppercase text-slate-500 dark:text-slate-400 mb-1">Cross-Validation</p>
                  <dl className="grid grid-cols-2 gap-1 text-[9px] sm:text-[10px] text-slate-700 dark:text-slate-300">
                    <div>
                      <span className="text-slate-500">RMSE:</span>
                      <span className="ml-1 font-semibold">{formatNumber(validationData.data.validation_results.cross_validation.RMSE ?? null)}</span>
                    </div>
                    <div>
                      <span className="text-slate-500">MAPE:</span>
                      <span className="ml-1 font-semibold">{formatPercent(validationData.data.validation_results.cross_validation.MAPE ?? null)}</span>
                    </div>
                    {validationData.data.validation_results.cross_validation.std_RMSE && (
                      <div className="col-span-2 text-[8px] sm:text-[9px] text-slate-500">
                        Std: RMSE {formatNumber(validationData.data.validation_results.cross_validation.std_RMSE)}
                      </div>
                    )}
                  </dl>
                </div>
              )}
            </div>
          )}
        </aside>
      </div>
    </section>
  );
}

interface MetricItemProps {
  label: string;
  value: string;
}

function MetricItem({ label, value }: MetricItemProps) {
  return (
    <div>
      <p className="text-[9px] sm:text-[11px] uppercase tracking-wider text-slate-600 dark:text-slate-500">{label}</p>
      <p className="text-xs sm:text-sm font-semibold text-slate-900 dark:text-white">{value}</p>
    </div>
  );
}

