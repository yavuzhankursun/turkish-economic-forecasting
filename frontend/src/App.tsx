import { useMemo, useState, useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { fetchComprehensiveAnalysis, fetchLatestComprehensiveAnalysis, fetchModelValidation } from './api';
import type { ChartSeriesPoint, IndicatorForecast, NewsAnalysis } from './types';
import type { ValidationResponse } from './api';
import { Toolbar } from './components/Toolbar';
import { IndicatorCard } from './components/IndicatorCard';
import { ChartPanel } from './components/ChartPanel';
import { NewsSummary } from './components/NewsSummary';
import { LoadingState } from './components/LoadingState';
import { ErrorState } from './components/ErrorState';

function mapSeries(series?: Record<string, number>): ChartSeriesPoint[] {
  if (!series) return [];
  return Object.entries(series)
    .map(([time, value]) => ({ time, value }))
    .sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime());
}

const INDICATOR_LABELS: Record<string, { title: string; currency?: boolean }> = {
  usd_try: { title: 'USD/TRY Kuru', currency: true },
  inflation: { title: 'TÜFE Enflasyon Oranı' },
  interest_rate: { title: 'Politika Faizi (%)' }
};

function getNewsSummary(indicatorKey: keyof typeof INDICATOR_LABELS, text?: string | null) {
  if (!text) return undefined;
  if (indicatorKey === 'usd_try') return `Döviz masası: ${text}`;
  if (indicatorKey === 'inflation') return `Enflasyon görünümü: ${text}`;
  return `Para politikası yorumu: ${text}`;
}

export default function App() {
  const [lastUpdated, setLastUpdated] = useState<string | undefined>();
  const queryClient = useQueryClient();

  // Sayfa yüklendiğinde kaydedilmiş verileri getir
  const { data: cachedData, isLoading: isLoadingCached } = useQuery({
    queryKey: ['latest-analysis'],
    queryFn: fetchLatestComprehensiveAnalysis,
    staleTime: Infinity, // Cache'i manuel olarak güncelle
    gcTime: Infinity, // Cache'i silme
    retry: false // 404 hatasında tekrar deneme
  });

  // Yeni analiz yapma mutation'ı
  const { mutate, data: newData, isPending, isError, error } = useMutation({
    mutationFn: fetchComprehensiveAnalysis,
    onSuccess: (response) => {
      const timestamp = response.data?.news_analysis?.timestamp ?? new Date().toISOString();
      setLastUpdated(timestamp);
      // Cache'i yeni verilerle güncelle
      queryClient.setQueryData(['latest-analysis'], response);
      // Validasyon sorgularını invalidate et ki fresh data çeksin
      queryClient.invalidateQueries({ queryKey: ['validation'] });
    }
  });

  // Güncelleme butonuna basıldığında yeni analiz yap
  const handleUpdate = () => {
    // Önce validasyon cache'ini temizle
    queryClient.invalidateQueries({ queryKey: ['validation'] });
    mutate();
  };

  // Hangi veriyi kullanacağımızı belirle (yeni veri varsa onu kullan, yoksa cache'den al)
  // cachedData null olabilir (404 durumunda)
  const data = newData || (cachedData && cachedData.success ? cachedData : null);

  // lastUpdated'i ayarla
  useEffect(() => {
    if (data?.data) {
      const timestamp = data.data?.news_analysis?.timestamp ?? data.updated_at ?? data.created_at ?? new Date().toISOString();
      setLastUpdated(timestamp);
    }
  }, [data]);

  const indicators = useMemo(() => {
    const result = data?.data;
    if (!result) return {} as Record<string, IndicatorForecast>;
    return {
      usd_try: result.usd_try,
      inflation: result.inflation,
      interest_rate: result.interest_rate
    };
  }, [data]);

  // Her gösterge için validasyon sorguları - Sadece bir kez çalıştır, cache kullan
  const usdValidation = useQuery({
    queryKey: ['validation', 'usd_try'],
    queryFn: () => fetchModelValidation({ indicator_type: 'usd_try', validation_type: 'both' }),
    enabled: !!data?.data?.usd_try,
    staleTime: 5 * 60 * 1000, // 5 dakika cache
    gcTime: 10 * 60 * 1000, // 10 dakika garbage collection
    refetchOnMount: false, // Mount'ta tekrar çalıştırma
    refetchOnWindowFocus: false, // Window focus'ta tekrar çalıştırma
    refetchOnReconnect: false, // Reconnect'te tekrar çalıştırma
    retry: 1
  });

  const inflationValidation = useQuery({
    queryKey: ['validation', 'inflation'],
    queryFn: () => fetchModelValidation({ indicator_type: 'inflation', validation_type: 'both' }),
    enabled: !!data?.data?.inflation,
    staleTime: 5 * 60 * 1000, // 5 dakika cache
    gcTime: 10 * 60 * 1000, // 10 dakika garbage collection
    refetchOnMount: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    retry: 1
  });

  const interestRateValidation = useQuery({
    queryKey: ['validation', 'interest_rate'],
    queryFn: () => fetchModelValidation({ indicator_type: 'interest_rate', validation_type: 'both' }),
    enabled: !!data?.data?.interest_rate,
    staleTime: 5 * 60 * 1000, // 5 dakika cache
    gcTime: 10 * 60 * 1000, // 10 dakika garbage collection
    refetchOnMount: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    retry: 1
  });

  const validationMap: Record<string, ValidationResponse | null> = {
    usd_try: usdValidation.data ?? null,
    inflation: inflationValidation.data ?? null,
    interest_rate: interestRateValidation.data ?? null
  };

  const reportDate = lastUpdated ? new Date(lastUpdated).toLocaleString('tr-TR') : new Date().toLocaleString('tr-TR');

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-blue-50 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-4 sm:gap-6 md:gap-8 px-3 py-4 sm:px-4 sm:py-6 md:px-6 md:py-10">
        {/* Yazdırmada görünen başlık (dark mode) */}
        <div className="hidden print:block print:py-4 print:pb-6 print:text-slate-100">
          <h1 className="text-xl sm:text-2xl font-bold">Ekonomik Göstergeler Dashboard</h1>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">{reportDate}</p>
        </div>
        <Toolbar onRun={handleUpdate} loading={isPending || isLoadingCached} lastUpdated={lastUpdated} />

        {isError && <ErrorState message={(error as Error)?.message ?? 'Bilinmeyen hata'} />}

        {!data && (isPending || isLoadingCached) ? (
          <LoadingState />
        ) : !data && !isPending && !isLoadingCached ? (
          <div className="rounded-2xl sm:rounded-3xl border border-blue-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/80 p-4 sm:p-6 md:p-8 text-center">
            <p className="text-sm sm:text-base text-slate-600 dark:text-slate-400 mb-4">
              Henüz analiz yapılmamış. "Güncelle" butonuna basarak ilk analizi başlatabilirsiniz.
            </p>
          </div>
        ) : (
          <>
            <div className="flex flex-col gap-6">
              {(['usd_try', 'inflation', 'interest_rate'] as const).map((key) => {
                const indicator = indicators[key];
                const label = INDICATOR_LABELS[key];
                const chartHistorical = mapSeries(indicator?.historical);
                const chartPrimary = mapSeries(indicator?.forecast);
                const chartEnhanced = mapSeries(indicator?.enhanced_forecast);
                const newsData = data?.data?.news_analysis;
                const analysisItem = newsData ? newsData[key as 'usd_try' | 'inflation' | 'interest_rate'] : undefined;
                const newsMultiplier = analysisItem?.multiplier ?? indicator?.news_multiplier;
                const analysisText = analysisItem?.analysis;
                const newsSummary = getNewsSummary(key, analysisText) ?? analysisText ?? undefined;

                return (
                  <IndicatorCard
                    key={key}
                    title={label.title}
                    currency={label.currency}
                    indicator={indicator ?? null}
                    newsMultiplier={newsMultiplier}
                    newsSummary={newsSummary}
                    validationData={validationMap[key]}
                    chart={<ChartPanel historical={chartHistorical} primary={chartPrimary} secondary={chartEnhanced} />}
                  />
                );
              })}
            </div>

            <NewsSummary data={data?.data?.news_analysis} loading={isPending} />
          </>
        )}
      </div>
    </div>
  );
}

