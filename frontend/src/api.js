const API_BASE = import.meta.env.VITE_API_BASE || ''

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    const detail = data.detail || data.error || res.statusText
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return data
}

export const api = {
  health: () => request('/health'),
  home: () => request('/api/home'),
  suggest: (q) => request(`/api/suggest?q=${encodeURIComponent(q)}`),
  models: () => request('/api/models'),
  modelStatus: (ticker) => request(`/api/model/${encodeURIComponent(ticker)}`),
  deleteModel: (ticker, market) => {
    const qs = market ? `?market=${encodeURIComponent(market)}` : ''
    return request(`/api/model/${encodeURIComponent(ticker)}${qs}`, {
      method: 'DELETE',
    })
  },
  train: (ticker) =>
    request('/api/train', {
      method: 'POST',
      body: JSON.stringify({ ticker }),
    }),
  trainStatus: (jobId) => request(`/api/train/${jobId}`),
  predict: (ticker) => request(`/api/predict/${encodeURIComponent(ticker)}`),
  tickers: () => request('/api/tickers'),
}
