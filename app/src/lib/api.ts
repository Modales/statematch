import type { Architecture, AuditEntry, Case, EvidenceItem, Finding, Health, Metrics, Summary } from '@/types'

const base = '/api'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${base}${path}`, {
    ...options,
    headers: options?.body instanceof FormData ? options.headers : { 'Content-Type': 'application/json', ...options?.headers },
  })
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try {
      const body = await response.json()
      message = body.detail || message
    } catch { /* response was not JSON */ }
    throw new Error(message)
  }
  return response.json()
}

export const api = {
  health: () => request<Health>('/health'),
  summary: () => request<Summary>('/dashboard/summary'),
  cases: (search = '', status = 'all') => request<{ items: Case[] }>(`/cases?search=${encodeURIComponent(search)}&status=${status}`),
  metrics: () => request<Metrics>('/model/metrics'),
  architecture: () => request<Architecture>('/model/architecture'),
  evidence: (id: string) => request<{ items: EvidenceItem[]; rules: Finding[]; method: string; caveat: string }>(`/cases/${encodeURIComponent(id)}/evidence`),
  audit: (id: string) => request<{ chain_valid: boolean; entries: AuditEntry[] }>(`/cases/${encodeURIComponent(id)}/audit`),
  action: (id: string, action: string) => request<{ case: Case; receipt: AuditEntry }>(`/cases/${encodeURIComponent(id)}/actions`, {
    method: 'POST', body: JSON.stringify({ action, actor: 'demo.analyst' }),
  }),
  inject: (scenario: string) => request<Case>('/demo/inject', { method: 'POST', body: JSON.stringify({ scenario }) }),
  ingest: (file: File) => {
    const body = new FormData()
    body.append('file', file)
    return request<{ imported: string[]; errors: string[]; count: number }>('/ingest', { method: 'POST', body })
  },
}
