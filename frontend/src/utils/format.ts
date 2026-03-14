export function formatNumber(value: number | null | undefined, opts: Intl.NumberFormatOptions = {}): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '—';
  }
  return new Intl.NumberFormat('tr-TR', {
    maximumFractionDigits: 2,
    ...opts
  }).format(value);
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '—';
  }
  return `${value.toFixed(2)}%`;
}

export function formatDate(label: string): string {
  const date = new Date(label);
  if (Number.isNaN(date.getTime())) {
    return label;
  }
  return new Intl.DateTimeFormat('tr-TR', {
    year: 'numeric',
    month: 'short'
  }).format(date);
}

