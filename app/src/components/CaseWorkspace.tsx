import { useEffect, useState } from 'react'
import { Check, Download, FileSearch, GitCompareArrows, Hand, LoaderCircle, RotateCcw, Send, ShieldAlert, UserCheck } from 'lucide-react'
import { api } from '@/lib/api'
import type { AuditEntry, Case, EvidenceItem, Finding } from '@/types'
import { StatusMark } from './StatusMark'
import { formatDate, formatMoney } from '@/lib/format'

type Tab = 'trace' | 'matches' | 'evidence' | 'audit'

export function CaseWorkspace({ item, onAction, actionPending }: { item: Case; onAction: (id: string, action: string) => void; actionPending: boolean }) {
  const [tab, setTab] = useState<Tab>('trace')
  const [evidence, setEvidence] = useState<EvidenceItem[]>([])
  const [evidenceMeta, setEvidenceMeta] = useState({ method: '', caveat: '' })
  const [audit, setAudit] = useState<{ chain_valid: boolean; entries: AuditEntry[] } | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const critical = item.rule_findings.some(finding => finding.severity === 'critical')
  const flagged = new Set(item.rule_findings.flatMap(finding => finding.event_ids))

  useEffect(() => {
    if (tab !== 'evidence' && tab !== 'audit') return
    let active = true
    const task = tab === 'evidence' ? api.evidence(item.id) : api.audit(item.id)
    task.then(result => {
      if (!active) return
      if (tab === 'evidence') {
        const value = result as Awaited<ReturnType<typeof api.evidence>>
        setEvidence(value.items); setEvidenceMeta({ method: value.method, caveat: value.caveat })
      } else setAudit(result as Awaited<ReturnType<typeof api.audit>>)
    }).catch(reason => { if (active) setError(reason.message) }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [item.id, tab, item.updated_at])

  const changeTab = (next: Tab) => {
    setTab(next)
    setError('')
    if (next === 'evidence') { setEvidence([]); setLoading(true) }
    if (next === 'audit') { setAudit(null); setLoading(true) }
  }

  const maxDelta = Math.max(...evidence.map(row => Math.abs(row.delta)), 0.001)
  return (
    <section className="case-workspace" aria-label={`Case ${item.id}`}>
      <header className="case-heading">
        <div>
          <div className="case-heading-line"><h1 className="mono">{item.id}</h1><StatusMark status={item.status} /></div>
          <p>{item.vendor_name} <span>·</span> {item.category} <span>·</span> {item.spend_area}</p>
          <p className="source-line">{item.source}</p>
        </div>
        <div className="case-amount"><span>Document value</span><strong>{formatMoney(item.amount)}</strong><small>EUR · anonymized source values</small></div>
      </header>

      <div className="decision-strip">
        <div><span>Mamba sequence risk</span><strong>{Math.round(item.model_risk * 100)}<small>%</small></strong><div className="risk-track"><i style={{ width: `${item.model_risk * 100}%` }} /></div></div>
        <div><span>Next likely activity</span><strong className="next-activity">{item.predicted_next}</strong><small>{(item.next_confidence * 100).toFixed(1)}% model probability</small></div>
        <div><span>Sequence NLL</span><strong>{item.model_nll.toFixed(3)}</strong><small>Lower is less surprising</small></div>
        <div><span>Control findings</span><strong>{item.rule_findings.length.toString().padStart(2, '0')}</strong><small>{critical ? 'Hard stop enforced' : 'No critical hard stop'}</small></div>
      </div>

      {item.rule_findings.length > 0 && <div className="findings-list">{item.rule_findings.map(finding => <FindingRow key={`${finding.code}-${finding.event_ids.join('-')}`} finding={finding} />)}</div>}

      <nav className="workspace-tabs" aria-label="Case details">
        {(['trace', 'matches', 'evidence', 'audit'] as Tab[]).map(value => <button key={value} className={tab === value ? 'active' : ''} onClick={() => changeTab(value)}>
          {value === 'trace' ? `Event trace · ${item.events.length}` : value === 'matches' ? `Match ledger · ${item.links.length}` : value === 'evidence' ? 'Replay evidence' : 'Audit chain'}
        </button>)}
      </nav>

      <div className="workspace-content">
        {tab === 'trace' && <div className="event-trace">
          <div className="event-header"><span>Time</span><span>Activity</span><span>Actor</span><span>Value</span><span>Surprise</span></div>
          {item.events.map((event, index) => <div className={`event-row ${flagged.has(event.id) ? 'flagged' : ''}`} key={event.id}>
            <time>{formatDate(event.timestamp)}</time>
            <div className="event-activity"><span className="event-index mono">{String(index + 1).padStart(2, '0')}</span><strong>{event.activity}</strong><small className="mono">{event.id}</small></div>
            <span className="mono event-user">{event.user}</span>
            <span className="mono event-value">{formatMoney(event.amount)}</span>
            <span className="surprise-cell"><i style={{ width: `${Math.min(100, (event.surprise || 0) * 18)}%` }} /><b className="mono">{event.surprise?.toFixed(2) ?? '—'}</b></span>
          </div>)}
        </div>}

        {tab === 'matches' && <div className="match-ledger">
          <div className="section-note"><GitCompareArrows size={17} /><p>Candidate associations combine Mamba hidden-state similarity, amount compatibility, and temporal decay. Hard controls still determine whether a match can be approved.</p></div>
          {item.links.length > 0 ? <div className="data-table match-table"><div className="data-table-head"><span>Goods receipt</span><span>Invoice</span><span>Link score</span></div>{item.links.map(link => <div className="data-table-row" key={`${link.receipt_id}-${link.invoice_id}`}><span className="mono">{link.receipt_id}</span><span className="mono">{link.invoice_id}</span><span><b className="mono">{(link.score * 100).toFixed(1)}%</b><i className="link-meter"><i style={{ width: `${link.score * 100}%` }} /></i></span></div>)}</div> : <div className="empty-state"><GitCompareArrows /><strong>No receipt–invoice pairs</strong><p>This trace does not contain both event types required for a link proposal.</p></div>}
        </div>}

        {tab === 'evidence' && <div className="evidence-view">
          <div className="section-note"><FileSearch size={17} /><div><p>{evidenceMeta.method}</p><small>{evidenceMeta.caveat}</small></div></div>
          {loading && <Loading label="Replaying event sequences through Mamba" />}
          {error && <ErrorMessage message={error} />}
          {!loading && evidence.map((row, index) => <div className="evidence-row" key={row.event_id}><span className="mono">{String(index + 1).padStart(2, '0')}</span><div><strong>{row.activity}</strong><small className="mono">{row.event_id}</small></div><div className="evidence-bar"><i className={row.delta >= 0 ? 'positive' : 'negative'} style={{ width: `${Math.abs(row.delta) / maxDelta * 100}%` }} /></div><b className="mono">{row.delta >= 0 ? '+' : ''}{row.delta.toFixed(3)}</b></div>)}
        </div>}

        {tab === 'audit' && <div className="audit-view">
          <div className="audit-heading"><div><span>SHA-256 hash chain</span><strong>{audit?.chain_valid ? 'Verified' : 'Verification pending'}</strong></div><div className="download-actions"><a className="button secondary" href={`/api/cases/${encodeURIComponent(item.id)}/audit?format=download`}><Download size={14} />JSON</a><a className="button secondary" href={`/api/cases/${encodeURIComponent(item.id)}/audit?format=csv`}><Download size={14} />CSV</a></div></div>
          {loading && <Loading label="Verifying audit entries" />}
          {error && <ErrorMessage message={error} />}
          {audit?.entries.slice().reverse().map(entry => <div className="audit-entry" key={entry.id}><div><span className={`audit-valid ${entry.valid ? 'valid' : ''}`}>{entry.valid ? 'VALID' : 'INVALID'}</span><strong>{entry.action}</strong><small>{entry.actor} · {formatDate(entry.created_at)}</small></div><p>{entry.note}</p><code>{entry.hash}</code></div>)}
        </div>}
      </div>

      <footer className="case-actions">
        <div><span>Workflow action</span><small>Every action persists and appends an audit receipt.</small></div>
        <div className="action-buttons">
          {['held', 'in_review', 'awaiting_receipt', 'reconciled'].includes(item.status) && <button className="button secondary" disabled={actionPending} onClick={() => onAction(item.id, 'reopen')}><RotateCcw size={15} />Reopen</button>}
          {!critical && item.status !== 'reconciled' && <button className="button primary" disabled={actionPending} onClick={() => onAction(item.id, 'approve_match')}><Check size={15} />Approve match</button>}
          {item.status !== 'held' && <button className="button secondary" disabled={actionPending} onClick={() => onAction(item.id, 'hold_clearance')}><Hand size={15} />Hold clearance</button>}
          <button className="button secondary" disabled={actionPending} onClick={() => onAction(item.id, 'request_receipt')}><Send size={15} />Request receipt</button>
          <button className="icon-button labeled" disabled={actionPending} onClick={() => onAction(item.id, 'assign_review')} title="Assign to operations analyst"><UserCheck size={17} /><span>Assign</span></button>
        </div>
      </footer>
    </section>
  )
}

function FindingRow({ finding }: { finding: Finding }) {
  return <div className={`finding ${finding.severity}`}><ShieldAlert size={17} /><div><strong>{finding.title}</strong><p>{finding.detail}</p><code>{finding.code}</code></div></div>
}

function Loading({ label }: { label: string }) {
  return <div className="loading-state"><LoaderCircle className="spin" size={20} /><span>{label}</span></div>
}

function ErrorMessage({ message }: { message: string }) {
  return <div className="error-state" role="alert">{message}</div>
}
