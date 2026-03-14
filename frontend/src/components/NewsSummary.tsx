import React from 'react';
import type { NewsAnalysis } from '../types';
import { formatDate } from '../utils/format';

interface NewsSummaryProps {
  data?: NewsAnalysis;
  loading?: boolean;
}

export function NewsSummary({ data, loading }: NewsSummaryProps) {
  if (loading) {
    return (
      <aside className="print-report-block rounded-2xl sm:rounded-3xl border border-blue-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/80 p-4 sm:p-6">
        <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400">Haber analizleri yükleniyor...</p>
      </aside>
    );
  }

  if (!data) {
    return (
      <aside className="print-report-block rounded-2xl sm:rounded-3xl border border-blue-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/80 p-4 sm:p-6">
        <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400">
          Haber analizi henüz oluşturulmadı. Analizi çalıştırarak güncel ekonomik yorumları alabilirsiniz.
        </p>
      </aside>
    );
  }

  const timestamp = data.timestamp ? formatDate(data.timestamp) : '—';
  return (
    <aside className="print-report-block rounded-2xl sm:rounded-3xl border border-blue-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/80 p-4 sm:p-6 flex flex-col gap-3 sm:gap-4">
      <div>
        <p className="text-[10px] sm:text-xs uppercase tracking-wider text-slate-600 dark:text-slate-500">Claude Sonnet 4.5 Haber Özeti</p>
        <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400">Son güncelleme: {timestamp}</p>
      </div>

      <NewsRow title="USD/TRY" item={data.usd_try} tone="teal" />
      <NewsRow title="Enflasyon" item={data.inflation} tone="amber" />
      <NewsRow title="Politika Faizi" item={data.interest_rate} tone="rose" />
    </aside>
  );
}

interface NewsRowProps {
  title: string;
  item?: NewsAnalysis['usd_try'];
  tone: 'teal' | 'amber' | 'rose';
}

function NewsRow({ title, item, tone }: NewsRowProps) {
  const badgeClass = {
    teal: 'bg-teal-100 dark:bg-teal-500/20 text-teal-700 dark:text-teal-300 border-teal-300 dark:border-teal-500/40',
    amber: 'bg-amber-100 dark:bg-amber-500/20 text-amber-700 dark:text-amber-300 border-amber-300 dark:border-amber-500/40',
    rose: 'bg-rose-100 dark:bg-rose-500/20 text-rose-700 dark:text-rose-300 border-rose-300 dark:border-rose-500/40'
  }[tone];

  return (
    <div className="rounded-xl sm:rounded-2xl border border-blue-200 dark:border-slate-800/80 bg-blue-50/50 dark:bg-slate-900/70 p-3 sm:p-4">
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 sm:gap-3">
        <div className="min-w-0 flex-1">
          <p className="text-[10px] sm:text-xs uppercase tracking-wide text-slate-600 dark:text-slate-500">{title}</p>
          <p className="text-xs sm:text-sm text-slate-700 dark:text-slate-300 leading-relaxed mt-1">
            {item?.analysis ?? 'Analiz bulunamadı.'}
          </p>
        </div>
        <span className={`text-[10px] sm:text-xs font-semibold px-2 py-1 sm:px-3 sm:py-1 rounded-full border ${badgeClass} self-start sm:self-auto flex-shrink-0`}>
          x{item?.multiplier?.toFixed(3) ?? '1.000'}
        </span>
      </div>
      <p className="text-[10px] sm:text-xs text-slate-600 dark:text-slate-500 mt-2">
        Güven skoru: {(item?.confidence ?? 0.5).toFixed(2)} · Haber sayısı: {item?.articles_analyzed ?? 0}
      </p>
    </div>
  );
}

