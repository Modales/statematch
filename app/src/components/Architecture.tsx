import { ArrowDown, ArrowUpRight, Braces, Check, Database, FileInput, GitCommitHorizontal, Layers3, ShieldCheck, X } from 'lucide-react'
import type { Architecture as ArchitectureData } from '@/types'

const icons = [FileInput, Layers3, Braces, ShieldCheck, GitCommitHorizontal]

export function Architecture({ data }: { data: ArchitectureData }) {
  return (
    <div className="architecture-page">
      <header className="page-title"><div><span className="overline">SYSTEM / ATTENTION-FREE BY CONSTRUCTION</span><h1>State, not tokens in a chat window.</h1><p>A process model produces scores. Deterministic controls make decisions. The workflow commits auditable work.</p></div><div className="architecture-badges"><span><Check size={14} />Mamba checkpoint</span><span><X size={14} />No attention</span><span><X size={14} />No LLM calls</span></div></header>
      <section className="pipeline">{data.stages.map((stage, index) => {
        const Icon = icons[index] || Database
        return <div className="pipeline-item" key={stage.id}><div className="pipeline-node"><span className="mono">{String(index + 1).padStart(2, '0')}</span><Icon size={22} /><div><h2>{stage.name}</h2><p>{stage.detail}</p></div></div>{index < data.stages.length - 1 && <ArrowDown className="pipeline-arrow" size={17} />}</div>
      })}</section>
      <div className="architecture-grid">
        <section className="code-section"><div className="section-heading"><div><span>Core recurrence</span><h2>Selective state update</h2></div><Braces size={19} /></div><pre><code>{`A = -exp(A_log)              # stable state matrix\nΔ, B, C = input_projections(x_t)\nstate_t = exp(Δ_t · A) · state_(t−1)\n          + Δ_t · B_t · x_t\ny_t = C_t · state_t + D · x_t\n\n# No QKᵀ matrix. No attention cache.\n# Portable PyTorch recurrence for CPU VMs.`}</code></pre><div className="config-grid">{Object.entries(data.config).map(([key, value]) => <div key={key}><span>{key}</span><strong className="mono">{value}</strong></div>)}</div></section>
        <section className="control-section"><div className="section-heading"><div><span>Decision boundary</span><h2>Model versus control</h2></div><ShieldCheck size={19} /></div><div className="boundary-row"><span>Next-event likelihood</span><strong>Mamba</strong></div><div className="boundary-row"><span>Token surprise + link representation</span><strong>Mamba</strong></div><div className="boundary-row"><span>Receipt-before-clearance</span><strong>Hard rule</strong></div><div className="boundary-row"><span>Duplicate invoice check</span><strong>Hard rule</strong></div><div className="boundary-row"><span>Critical-case approval</span><strong>Blocked</strong></div><div className="boundary-row"><span>Final workflow action</span><strong>Human + policy</strong></div><div className="boundary-row"><span>Audit integrity</span><strong>SHA-256 chain</strong></div></section>
      </div>
      <section className="research-section"><div><span className="overline">PRIMARY SOURCES</span><h2>Inspect the foundations.</h2></div><div className="research-links"><a href="https://arxiv.org/abs/2312.00752" target="_blank" rel="noreferrer"><span>Mamba: Linear-Time Sequence Modeling with Selective State Spaces</span><ArrowUpRight size={17} /></a><a href="https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1" target="_blank" rel="noreferrer"><span>BPI Challenge 2019 · official 4TU research dataset</span><ArrowUpRight size={17} /></a><a href="https://github.com/Modales/statematch" target="_blank" rel="noreferrer"><span>StateMatch source and reproducibility notes</span><ArrowUpRight size={17} /></a></div></section>
    </div>
  )
}
