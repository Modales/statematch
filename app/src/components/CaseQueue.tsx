import { Search } from 'lucide-react'
import type { Case } from '@/types'
import { StatusMark } from './StatusMark'
import { formatMoney } from '@/lib/format'

const filters = [
  ['all', 'All'], ['exception', 'Exceptions'], ['review', 'Review'], ['ready', 'Ready'], ['reconciled', 'Done'],
]

export function CaseQueue({ cases, selectedId, onSelect, search, onSearch, status, onStatus }: {
  cases: Case[]
  selectedId?: string
  onSelect: (id: string) => void
  search: string
  onSearch: (value: string) => void
  status: string
  onStatus: (value: string) => void
}) {
  return (
    <section className="case-queue" aria-label="Case queue">
      <div className="queue-tools">
        <label className="search-field">
          <Search size={15} aria-hidden="true" />
          <input id="case-search" value={search} onChange={event => onSearch(event.target.value)} placeholder="Case, vendor, activity…" aria-label="Search cases" />
          <kbd>/</kbd>
        </label>
        <div className="queue-filters" aria-label="Filter cases">
          {filters.map(([value, label]) => <button key={value} className={status === value ? 'active' : ''} onClick={() => onStatus(value)}>{label}</button>)}
        </div>
      </div>
      <div className="queue-column-labels"><span>Case / vendor</span><span>Value</span></div>
      <div className="queue-rows">
        {cases.map(item => (
          <button key={item.id} className={`queue-row ${selectedId === item.id ? 'selected' : ''}`} onClick={() => onSelect(item.id)}>
            <span className="queue-row-top"><span className="mono">{item.id}</span><span className="queue-value">{formatMoney(item.amount)}</span></span>
            <span className="queue-vendor">{item.vendor_name}</span>
            <span className="queue-row-bottom"><StatusMark status={item.status} /><span>{item.events.length} events</span><span>{Math.round(item.model_risk * 100)}% risk</span></span>
          </button>
        ))}
        {cases.length === 0 && <div className="queue-empty"><strong>No cases in this view</strong><span>Change the filter or search term.</span></div>}
      </div>
      <div className="queue-footer"><span>{cases.length} loaded cases</span><span>Live inference</span></div>
    </section>
  )
}
