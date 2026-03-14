export type MetricRecord = Record<string, number | string | null>;

export interface IndicatorMetrics {
  MAE?: number | null;
  MSE?: number | null;
  RMSE?: number | null;
  MAPE?: number | null;
  samples?: number;
  mode?: string;
}

export interface IndicatorForecast {
  status: string;
  data_points?: number;
  last_value?: number;
  feature_count?: number;
  forecast?: Record<string, number>;
  enhanced_forecast?: Record<string, number>;
  historical?: Record<string, number>;
  metrics?: IndicatorMetrics | null;
  news_multiplier?: number;
}

export interface NewsAnalysisItem {
  multiplier: number;
  analysis: string;
  confidence?: number;
  reasoning?: string;
  articles_analyzed?: number;
}

export interface NewsAnalysis {
  usd_try?: NewsAnalysisItem;
  inflation?: NewsAnalysisItem;
  interest_rate?: NewsAnalysisItem;
  timestamp?: string;
}

export interface ComprehensiveResponse {
  success: boolean;
  data: {
    usd_try: IndicatorForecast;
    inflation: IndicatorForecast;
    interest_rate: IndicatorForecast;
    news_analysis?: NewsAnalysis;
    performance?: Record<string, unknown>;
  } | null;
  message: string;
  timestamp?: string;
  created_at?: string;
  updated_at?: string;
}

export interface ChartSeriesPoint {
  time: string;
  value: number;
}

