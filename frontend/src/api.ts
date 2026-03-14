import type { ComprehensiveResponse } from './types';

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? window.location.origin;

export async function fetchComprehensiveAnalysis(): Promise<ComprehensiveResponse> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 600000); // 10 dakika timeout

  try {
    const response = await fetch(`${BASE_URL.replace(/\/$/, '')}/api/multi/comprehensive`, {
      headers: {
        'Content-Type': 'application/json',
        'Connection': 'keep-alive'
      },
      signal: controller.signal,
      keepalive: true
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`API hatası (${response.status}): ${text}`);
    }

    return response.json();
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('İstek zaman aşımına uğradı. Analiz çok uzun sürdü, lütfen tekrar deneyin.');
    }
    throw error;
  }
}

export async function fetchLatestComprehensiveAnalysis(): Promise<ComprehensiveResponse | null> {
  try {
    const response = await fetch(`${BASE_URL.replace(/\/$/, '')}/api/multi/comprehensive/latest`, {
      headers: {
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      if (response.status === 404) {
        // Kaydedilmiş analiz yok, null döndür
        return null;
      }
      const text = await response.text();
      throw new Error(`API hatası (${response.status}): ${text}`);
    }

    return response.json();
  } catch (error) {
    if (error instanceof Error && (error.message.includes('404') || error.message.includes('Not Found'))) {
      return null;
    }
    // Diğer hatalar için null döndür (sessizce başarısız ol)
    console.warn('Kaydedilmiş analiz yüklenemedi:', error);
    return null;
  }
}

export interface ValidationRequest {
  indicator_type: 'usd_try' | 'inflation' | 'interest_rate';
  validation_type?: 'holdout' | 'crossval' | 'both';
  test_size?: number;
  n_splits?: number;
}

export interface ValidationResponse {
  success: boolean;
  data: {
    indicator_type: string;
    validation_results: {
      hold_out?: {
        MAE?: number;
        MSE?: number;
        RMSE?: number;
        MAPE?: number;
        train_size?: number;
        test_size?: number;
        error?: string;
      };
      cross_validation?: {
        MAE?: number;
        MSE?: number;
        RMSE?: number;
        MAPE?: number;
        std_RMSE?: number;
        std_MAPE?: number;
        n_splits?: number;
        error?: string;
      };
    };
    data_points?: number;
  };
  message: string;
  timestamp?: string;
}

export async function fetchModelValidation(params: ValidationRequest): Promise<ValidationResponse> {
  const response = await fetch(`${BASE_URL.replace(/\/$/, '')}/api/model/validate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(params)
  });

  const body = await response.json().catch(() => ({} as { message?: string }));

  if (!response.ok) {
    if (response.status === 404) {
      return {
        success: false,
        data: {
          indicator_type: params.indicator_type,
          validation_results: {}
        },
        message: (body as { message?: string }).message ?? 'Veri bulunamadı. Veritabanı bağlantısını kontrol edin.'
      };
    }
    const text = typeof body === 'object' && body !== null && 'message' in body ? (body as { message: string }).message : JSON.stringify(body);
    throw new Error(`API hatası (${response.status}): ${text}`);
  }

  return body as ValidationResponse;
}

