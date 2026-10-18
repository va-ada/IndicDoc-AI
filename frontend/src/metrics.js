// Formats a metric for display. Missing or non-numeric values show as a dash,
// never as a made-up placeholder number.
export const formatMetric = (value, suffix = '') =>
  typeof value === 'number' && Number.isFinite(value) ? `${value}${suffix}` : '—'
