import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'
import { Activity, Beaker, Check, ChevronRight, CircleAlert, FileUp, GitBranch, Menu, Play, Plus, ShieldCheck, X } from 'lucide-react'
import './App.css'
import { Architecture } from '@/components/Architecture'
import { CaseQueue } from '@/components/CaseQueue'
import { CaseWorkspace } from '@/components/CaseWorkspace'
import { Modal } from '@/components/Modal'

const ModelLab = lazy(() => import('@/components/ModelLab').then(module => ({ default: module.ModelLab })))
import { formatMoney } from '@/lib/format'
import { api } from '@/lib/api'
import type { Architecture as ArchitectureData, Case, Health, Metrics, Summary } from '@/types'

type View = 'operations' | 'lab' | 'architecture'

export default function App() {
  const [view, setView] = useState<View>('operations')
  const [cases, setCases] = useState<Case[]>([])
  const [selectedId, setSelectedId] = useState<string>()
  const [summary, setSummary] = useState<Summary>()
  const [health, setHealth] = useState<Health>()
  const [metrics, setMetrics] = useState<Metrics>()
  const [architecture, setArchitecture] = useState<ArchitectureData>()
  const [search, setSearch] = useState('')
  const [status, setStatus] = useState('all')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [actionPending, setActionPending] = useState(false)
  const [uploadOpen, setUploadOpen] = useState(false)
  const [demoOpen, setDemoOpen] = useState(false)
  const [mobileNav, setMobileNav] = useState(false)
  const [file, setFile] = useState<File>()
  const [uploadResult, setUploadResult] = useState<{ imported: string[]; errors: string[]; count: number }>()
  const [toast, setToast] = useState('')
  const mounted = useRef(false)

  const loadQueue = useCallback(async () => {
    try {
      const [queue, totals] = await Promise.all([api.cases(search, status), api.summary()])
      setCases(queue.items)
      setSummary(totals)
      setError('')
      setSelectedId(current => queue.items.some(item => item.id === current) ? current : queue.items.find(item => item.status === 'exception')?.id || queue.items[0]?.id)
    } catch (reason) { setError(errorMessage(reason) || 'Could not load StateMatch') }
    finally { setLoading(false) }
  }, [search, status])

  useEffect(() => {
    const timer = window.setTimeout(loadQueue, mounted.current ? 180 : 0)
    mounted.current = true
    return () => window.clearTimeout(timer)
  }, [loadQueue])

  useEffect(() => {
    Promise.all([api.health(), api.metrics(), api.architecture()]).then(([healthValue, metricsValue, architectureValue]) => {
      setHealth(healthValue); setMetrics(metricsValue); setArchitecture(architectureValue)
    }).catch(reason => setError(errorMessage(reason)))
  }, [])

  useEffect(() => {
    const keydown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement
      if (event.key === '/' && !['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) {
        event.preventDefault(); setView('operations'); window.setTimeout(() => document.getElementById('case-search')?.focus(), 0)
      }
    }
    document.addEventListener('keydown', keydown)
    return () => document.removeEventListener('keydown', keydown)
  }, [])

  useEffect(() => {
    if (!toast) return
    const timer = window.setTimeout(() => setToast(''), 4200)
    return () => window.clearTimeout(timer)
  }, [toast])

  const selected = cases.find(item => item.id === selectedId)
  const switchView = (next: View) => { setView(next); setMobileNav(false) }

  const runAction = async (id: string, action: string) => {
    setActionPending(true)
    try {
      const result = await api.action(id, action)
      setCases(current => current.map(item => item.id === id ? result.case : item))
      setSummary(await api.summary())
      setToast(`${action.replaceAll('_', ' ')} committed · receipt ${result.receipt.hash.slice(0, 12)}…`)
    } catch (reason) { setToast(errorMessage(reason)) }
    finally { setActionPending(false) }
  }

  const inject = async (scenario: string) => {
    setActionPending(true)
    try {
      const item = await api.inject(scenario)
      setSearch(''); setStatus('all')
      const [queue, totals] = await Promise.all([api.cases(), api.summary()])
      setCases(queue.items); setSummary(totals); setSelectedId(item.id); setDemoOpen(false); setView('operations')
      setToast(`Live ${scenario.replaceAll('_', ' ')} case scored and added to the queue`)
    } catch (reason) { setToast(errorMessage(reason)) }
    finally { setActionPending(false) }
  }

  const upload = async () => {
    if (!file) return
    setActionPending(true); setUploadResult(undefined)
    try {
      const result = await api.ingest(file)
      setUploadResult(result)
      const [queue, totals] = await Promise.all([api.cases(), api.summary()])
      setCases(queue.items); setSummary(totals); setSearch(''); setStatus('all')
      if (result.imported[0]) { setSelectedId(result.imported[0]); setView('operations') }
      setToast(`${result.count} case${result.count === 1 ? '' : 's'} scored through Mamba`)
    } catch (reason) { setToast(errorMessage(reason)) }
    finally { setActionPending(false) }
  }

  const downloadExample = () => {
    const csv = 'case_id,activity,timestamp,amount,user\nDEMO-1001,Create Purchase Order Item,2026-01-01T09:00:00Z,1250,vendor_17\nDEMO-1001,Record Goods Receipt,2026-01-02T09:00:00Z,1250,user_08\nDEMO-1001,Record Invoice Receipt,2026-01-03T09:00:00Z,1250,user_11\nDEMO-1001,Clear Invoice,2026-01-04T09:00:00Z,1250,user_12\n'
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'statematch-example.csv'; anchor.click(); URL.revokeObjectURL(url)
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to workspace</a>
      <aside className={`app-sidebar ${mobileNav ? 'mobile-open' : ''}`}>
        <div className="brand"><div className="brand-mark" aria-hidden="true"><i /><i /><i /></div><div><strong>STATE<span>/</span>MATCH</strong><small>PROCESS CONTROL</small></div></div>
        <button className="mobile-close icon-button" onClick={() => setMobileNav(false)} aria-label="Close navigation"><X size={18} /></button>
        <nav className="primary-nav" aria-label="Primary navigation">
          <span>Workspace</span>
          <button className={view === 'operations' ? 'active' : ''} onClick={() => switchView('operations')}><Activity size={17} /><span>Operations</span><b>{summary?.exception_count ?? '—'}</b></button>
          <button className={view === 'lab' ? 'active' : ''} onClick={() => switchView('lab')}><Beaker size={17} /><span>Model lab</span><ChevronRight size={14} /></button>
          <button className={view === 'architecture' ? 'active' : ''} onClick={() => switchView('architecture')}><GitBranch size={17} /><span>Architecture</span><ChevronRight size={14} /></button>
        </nav>
        <div className="sidebar-process"><span>Control path</span>{['Ingest trace', 'Score sequence', 'Apply controls', 'Commit action'].map((label, index) => <div key={label}><i className={index < 3 ? 'complete' : ''}>{index < 3 ? <Check size={11} /> : index + 1}</i><b>{label}</b></div>)}</div>
        <div className="sidebar-foot"><div className="model-online"><i /><span>{health ? 'Mamba online' : 'Loading model'}</span></div><p>{health?.checkpoint || 'Portable checkpoint'}<br />CPU inference · no LLM</p><small>DATASET / BPI 2019<br />TEMPORAL HOLDOUT</small></div>
      </aside>
      {mobileNav && <button className="nav-scrim" aria-label="Close navigation" onClick={() => setMobileNav(false)} />}

      <div className="app-main">
        <header className="topbar">
          <button className="mobile-menu icon-button" onClick={() => setMobileNav(true)} aria-label="Open navigation"><Menu size={19} /></button>
          <div className="breadcrumb"><span>Procure-to-pay</span><ChevronRight size={13} /><strong>{view === 'operations' ? 'Exception operations' : view === 'lab' ? 'Measured evaluation' : 'System architecture'}</strong></div>
          <div className="topbar-actions">
            <div className="runtime-status"><ShieldCheck size={15} /><span>Controls active</span></div>
            <button className="button secondary" onClick={() => setUploadOpen(true)}><FileUp size={15} /><span>Import CSV</span></button>
            <button className="button primary" onClick={() => setDemoOpen(true)}><Plus size={15} /><span>Inject exception</span></button>
          </div>
        </header>

        {error && <div className="global-error" role="alert"><CircleAlert size={17} /><span>{error}</span><button onClick={loadQueue}>Retry</button></div>}

        <main id="main-content">
          {view === 'operations' && <>
            <section className="summary-strip">
              <div><span>Open queue</span><strong>{summary?.total_cases ?? '—'}</strong><small>Real scored cases</small></div>
              <div><span>Exceptions</span><strong>{summary?.exception_count ?? '—'}</strong><small>Critical controls</small></div>
              <div><span>Ready to reconcile</span><strong>{summary?.ready_count ?? '—'}</strong><small>No critical finding</small></div>
              <div><span>Value under review</span><strong>{summary ? formatMoney(summary.value_under_review) : '—'}</strong><small>Anonymized EUR</small></div>
              <div><span>Workflow resolved</span><strong>{summary?.review_reduction_simulation ?? '—'}<em>%</em></strong><small>Queue simulation</small></div>
            </section>
            <div className="operations-layout">
              <CaseQueue cases={cases} selectedId={selectedId} onSelect={setSelectedId} search={search} onSearch={setSearch} status={status} onStatus={setStatus} />
              {selected ? <CaseWorkspace key={selected.id} item={selected} onAction={runAction} actionPending={actionPending} /> : <div className="workspace-empty">{loading ? 'Loading real model-scored cases…' : 'Select a case to begin.'}</div>}
            </div>
            <div className="metric-disclaimer">{summary?.metric_note}</div>
          </>}
          {view === 'lab' && <Suspense fallback={<PageLoading label="Loading measured experiment" />}>{metrics ? <ModelLab metrics={metrics} /> : <PageLoading label="Loading measured experiment" />}</Suspense>}
          {view === 'architecture' && (architecture ? <Architecture data={architecture} /> : <PageLoading label="Loading runtime architecture" />)}
        </main>
      </div>

      {uploadOpen && <Modal title="Import an event trace" description="Upload CSV. The API parses, sorts, scores, controls, and persists every valid case." onClose={() => { setUploadOpen(false); setUploadResult(undefined) }}>
        <form onSubmit={event => { event.preventDefault(); upload() }}>
          <label className="file-drop"><FileUp size={24} /><strong>{file ? file.name : 'Choose a CSV file'}</strong><span>Required: case_id, activity, timestamp. Optional: amount, user. Maximum 2.5 MB.</span><input type="file" accept=".csv,text/csv" onChange={event => setFile(event.target.files?.[0])} /></label>
          {uploadResult && <div className="upload-result"><strong>{uploadResult.count} cases imported</strong>{uploadResult.errors.map(message => <p key={message}>{message}</p>)}</div>}
          <div className="modal-actions"><button type="button" className="button secondary" onClick={downloadExample}>Download example</button><button className="button primary" disabled={!file || actionPending} type="submit">{actionPending ? 'Scoring…' : 'Import and score'}</button></div>
        </form>
      </Modal>}

      {demoOpen && <Modal title="Inject a controlled exception" description="Each scenario mutates a real BPI trace, runs live Mamba inference, and appends a new queue case." onClose={() => setDemoOpen(false)}>
        <div className="scenario-list">
          <button disabled={actionPending} onClick={() => inject('duplicate_invoice')}><span>01</span><div><strong>Duplicate invoice</strong><p>Repeat an invoice receipt within two hours and test the hard stop.</p></div><Play size={16} /></button>
          <button disabled={actionPending} onClick={() => inject('premature_unblock')}><span>02</span><div><strong>Premature unblock</strong><p>Remove a payment block before the first goods receipt.</p></div><Play size={16} /></button>
          <button disabled={actionPending} onClick={() => inject('clear_before_gr')}><span>03</span><div><strong>Clearance before receipt</strong><p>Reorder clearance ahead of goods receipt and inspect the evidence.</p></div><Play size={16} /></button>
        </div>
      </Modal>}
      {toast && <div className="toast" role="status"><Check size={16} /><span>{toast}</span><button onClick={() => setToast('')} aria-label="Dismiss notification"><X size={15} /></button></div>}
    </div>
  )
}

function errorMessage(reason: unknown) {
  return reason instanceof Error ? reason.message : String(reason)
}

function PageLoading({ label }: { label: string }) {
  return <div className="page-loading"><Activity size={22} /><span>{label}</span></div>
}
