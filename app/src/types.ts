export type CaseStatus = 'exception' | 'review' | 'ready' | 'open' | 'reconciled' | 'held' | 'in_review' | 'awaiting_receipt'

export interface TraceEvent {
  id: string
  activity: string
  timestamp: string
  amount: number
  user: string
  surprise: number | null
}

export interface Finding {
  code: string
  severity: 'critical' | 'warning'
  title: string
  detail: string
  event_ids: string[]
}

export interface MatchLink {
  receipt_id: string
  invoice_id: string
  receipt_activity: string
  invoice_activity: string
  score: number
  method: string
}

export interface Case {
  id: string
  purchase_document: string
  item: string
  vendor: string
  vendor_name: string
  category: string
  spend_area: string
  spend_class: string
  item_type: string
  amount: number
  status: CaseStatus
  source: string
  synthetic_deviation?: string
  events: TraceEvent[]
  rule_findings: Finding[]
  links: MatchLink[]
  model_nll: number
  sequence_score: number
  model_risk: number
  predicted_next: string
  next_confidence: number
  resolution?: { action: string; actor: string; note?: string; at: string } | null
  updated_at?: string
}

export interface Summary {
  total_cases: number
  statuses: Partial<Record<CaseStatus, number>>
  exception_count: number
  ready_count: number
  reconciled_count: number
  resolved_count: number
  value_under_review: number
  event_count: number
  review_reduction_simulation: number
  metric_note: string
}

export interface Health {
  status: string
  version: string
  model_loaded: boolean
  architecture: string
  attention: boolean
  llm: boolean
  device: string
  checkpoint: string
}

export interface ModelResult {
  nll: number
  accuracy: number
  macro_f1: number
  parameters: number
  training_seconds?: number
  synthetic_deviation_auroc?: number
  paired_cases?: number
  note?: string
  latency?: { length: number; ms: number }[]
}

export interface EvaluationSlice {
  cases: number
  tokens: number
  nll: number
  accuracy: number
  macro_f1: number
  unknown_events: number
  unknown_event_rate: number
  cases_with_unknown_activity: number
  first_event: string
  last_event: string
  name?: string
  nll_ci95?: number[]
  accuracy_ci95?: number[]
  macro_f1_ci95?: number[]
  original_reported_nll?: number
  nll_absolute_delta?: number
}

export interface DeviationResult {
  pairs: number
  auroc: number
  average_precision: number
  score_increase_rate: number
  median_score_delta: number
  auroc_ci95?: number[]
  score_increase_rate_ci95?: number[]
}

export interface ExtendedEvaluation {
  protocol: string
  checkpoint_sha256: string
  bootstrap: { method: string; repetitions: number; seed: number }
  anchor_800: EvaluationSlice
  full_temporal_test: EvaluationSlice
  later_period: EvaluationSlice
  later_windows: EvaluationSlice[]
  length_buckets: EvaluationSlice[]
  categories: EvaluationSlice[]
  deviations: DeviationResult & {
    by_kind: Record<string, DeviationResult>
    assignment: string
    score: string
    note: string
  }
  generated_at: string
}

export interface Metrics {
  dataset: { train: number; validation: number; test: number; split: string }
  config: Record<string, number>
  vocab_size: number
  device: string
  generated_at: string
  models: { mamba: ModelResult; transformer: ModelResult; markov: ModelResult }
  model: { name: string; checkpoint: string; attention: boolean; llm: boolean; deployment: string; implementation: string }
  extended_evaluation?: ExtendedEvaluation
}

export interface EvidenceItem {
  event_id: string
  activity: string
  delta: number
  baseline_score: number
  replay_score: number
}

export interface AuditEntry {
  id: string
  action: string
  actor: string
  note: string
  created_at: string
  previous_hash: string
  hash: string
  valid: boolean
  payload: Record<string, unknown>
}

export interface Architecture {
  name: string
  attention: boolean
  llm: boolean
  config: Record<string, number>
  vocab: { activities: string[]; categories: string[] }
  stages: { id: string; name: string; detail: string }[]
}
