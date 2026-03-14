import React from 'react';
import { FiDownload, FiRefreshCw, FiTrendingUp, FiSun, FiMoon } from 'react-icons/fi';
import { useTheme } from '../contexts/ThemeContext';

interface ToolbarProps {
  onRun: () => void;
  loading: boolean;
  lastUpdated?: string;
}

export function Toolbar({ onRun, loading, lastUpdated }: ToolbarProps) {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="no-print flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 sm:gap-4 rounded-2xl sm:rounded-3xl border border-blue-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/70 px-3 py-3 sm:px-6 sm:py-4">
      <div className="flex items-center gap-2 sm:gap-3 text-slate-700 dark:text-slate-300 min-w-0 flex-1">
        <div className="h-9 w-9 sm:h-11 sm:w-11 rounded-xl sm:rounded-2xl bg-blue-100 dark:bg-brand/20 text-blue-600 dark:text-brand-light flex items-center justify-center flex-shrink-0">
          <FiTrendingUp size={18} className="sm:w-5 sm:h-5" />
        </div>
        <div className="min-w-0 flex-1">
          <h1 className="text-base sm:text-lg font-semibold text-slate-900 dark:text-white truncate">Türkiye Makro Analiz Konsolu</h1>
          <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400 hidden sm:block">
            Claude Sonnet 4.5 haber sentiment destekli ARIMA tahminlerini tek tuşla üretin.
          </p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 sm:gap-3">
        <p className="text-[10px] sm:text-xs text-slate-500 hidden md:block">
          Son güncelleme: {lastUpdated ? new Date(lastUpdated).toLocaleString('tr-TR') : '—'}
        </p>
        <button
          onClick={toggleTheme}
          className="inline-flex items-center gap-1 sm:gap-2 rounded-xl sm:rounded-2xl border border-blue-200 dark:border-slate-700 px-2 py-2 sm:px-4 sm:py-3 text-xs sm:text-sm text-blue-700 dark:text-slate-200 hover:border-blue-400 dark:hover:border-slate-500 transition flex-shrink-0"
          title={theme === 'dark' ? 'Açık temaya geç' : 'Koyu temaya geç'}
        >
          {theme === 'dark' ? <FiSun size={16} className="sm:w-4 sm:h-4" /> : <FiMoon size={16} className="sm:w-4 sm:h-4" />}
          <span className="hidden sm:inline">{theme === 'dark' ? 'Açık Tema' : 'Koyu Tema'}</span>
        </button>
        <button
          onClick={onRun}
          disabled={loading}
          className="inline-flex items-center gap-1 sm:gap-2 rounded-xl sm:rounded-2xl bg-blue-600 dark:bg-brand px-3 py-2 sm:px-5 sm:py-3 text-xs sm:text-sm font-semibold text-white shadow-lg shadow-blue-600/30 dark:shadow-brand/30 transition hover:bg-blue-500 dark:hover:bg-brand-light disabled:opacity-60 flex-1 sm:flex-initial flex-shrink-0"
        >
          {loading ? (
            <>
              <FiRefreshCw className="animate-spin w-4 h-4 sm:w-4 sm:h-4" /> <span className="hidden sm:inline">Analiz çalışıyor...</span><span className="sm:hidden">Yükleniyor...</span>
            </>
          ) : (
            <>
              <FiRefreshCw className="w-4 h-4 sm:w-4 sm:h-4" /> <span className="hidden sm:inline">Güncelle</span><span className="sm:hidden">Güncelle</span>
            </>
          )}
        </button>
        <button
          onClick={() => window.print()}
          className="inline-flex items-center gap-1 sm:gap-2 rounded-xl sm:rounded-2xl border border-blue-200 dark:border-slate-700 px-2 py-2 sm:px-4 sm:py-3 text-xs sm:text-sm text-blue-700 dark:text-slate-200 hover:border-blue-400 dark:hover:border-slate-500 transition flex-shrink-0"
        >
          <FiDownload size={16} className="sm:w-4 sm:h-4" /> <span className="hidden sm:inline">Raporu Al</span>
        </button>
      </div>
    </div>
  );
}

