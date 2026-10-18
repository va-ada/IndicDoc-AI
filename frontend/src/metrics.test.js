import { test } from 'node:test'
import assert from 'node:assert/strict'
import { formatMetric } from './metrics.js'

test('formats real numbers, including zero', () => {
  assert.equal(formatMetric(0.8123), '0.8123')
  assert.equal(formatMetric(0), '0')
  assert.equal(formatMetric(1.61, 's'), '1.61s')
})

test('shows a dash instead of a number when the metric is missing', () => {
  for (const missing of [undefined, null, 'N/A', NaN, Infinity]) {
    assert.equal(formatMetric(missing), '—')
    assert.equal(formatMetric(missing, 's'), '—')
  }
})
