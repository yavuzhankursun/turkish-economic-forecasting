import React from 'react';

export function LoadingState() {
  return (
    <div className="rounded-2xl sm:rounded-3xl border border-slate-800 bg-slate-900/60 p-4 sm:p-6 text-center text-slate-300">
      <div className="mx-auto mb-2 sm:mb-3 h-8 w-8 sm:h-10 sm:w-10 animate-spin rounded-full border-2 border-slate-600 border-t-brand-light" />
      <p className="text-xs sm:text-sm px-2">Sonnet 4.5 haber analizi ve ARIMA tahminleri hazırlanıyor...</p>
    </div>
  );
}

