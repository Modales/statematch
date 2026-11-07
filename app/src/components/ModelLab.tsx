import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Check, FlaskConical, Gauge, Scale } from 'lucide-react'
import type { Metrics } from '@/types'

export function ModelLab({ metrics }: { metrics: Metrics }) {
  const models = [
    { name: 'Mamba', ...metrics.models.mamba },
    { name: 'Transformer', ...metrics.models.transformer },
    { name: 'Markov', ...metrics.models.markov },
  ]
  const latency = metrics.models.mamba.latency || []
  return (
    <div className="lab-page">
      <header className="page-title"><div><span className="overline">MODEL LAB / MEASURED HOLDOUT</span><h1>Architecture has to earn its place.</h1><p>Real BPI 2019 traces. Temporal holdout. Same-width neural baselines. No fabricated benchmark values.</p></div><div className="experiment-stamp"><FlaskConical size={20} /><span>Experiment</span><strong>SM–019</strong><small>{new Date(metrics.generated_at).toLocaleString()}</small></div></header>
      <section className="experiment-protocol">
        <div><span>Training traces</span><strong>{metrics.dataset.train.toLocaleString()}</strong></div>
        <div><span>Validation traces</span><strong>{metrics.dataset.validation.toLocaleString()}</strong></div>
        <div><span>Test traces</span><strong>{metrics.dataset.test.toLocaleString()}</strong></div>
        <div><span>Activity vocabulary</span><strong>{metrics.vocab_size}</strong></div>
        <div><span>Runtime</span><strong>CPU / {metrics.device}</strong></div>
      </section>
      <div className="lab-grid">
        <section className="chart-section"><div className="section-heading"><div><span>Next-event negative log-likelihood</span><h2>Lower is better</h2></div><Scale size={19} /></div><div className="chart-frame" aria-label="Model NLL comparison"><ResponsiveContainer width="100%" height="100%"><BarChart data={models} margin={{ top: 12, right: 12, left: -16, bottom: 0 }}><CartesianGrid vertical={false} stroke="#deded8" /><XAxis dataKey="name" tickLine={false} axisLine={false} /><YAxis tickLine={false} axisLine={false} /><Tooltip cursor={{ fill: '#ecece7' }} contentStyle={{ border: '1px solid #c8c8c1', borderRadius: 2, background: '#f9f9f6' }} formatter={(value) => Number(value).toFixed(3)} /><Bar dataKey="nll" fill="#222220" barSize={42} isAnimationActive={false} /></BarChart></ResponsiveContainer></div><div className="chart-footnote">Best checkpoint selected by validation NLL. Test events include an end-of-sequence target.</div></section>
        <section className="chart-section"><div className="section-heading"><div><span>Portable Mamba inference</span><h2>Latency by sequence length</h2></div><Gauge size={19} /></div><div className="chart-frame" aria-label="Mamba latency chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={latency} margin={{ top: 12, right: 16, left: -12, bottom: 0 }}><CartesianGrid vertical={false} stroke="#deded8" /><XAxis dataKey="length" tickLine={false} axisLine={false} /><YAxis unit="ms" tickLine={false} axisLine={false} /><Tooltip contentStyle={{ border: '1px solid #c8c8c1', borderRadius: 2, background: '#f9f9f6' }} formatter={(value) => `${Number(value).toFixed(2)} ms`} /><Line type="monotone" dataKey="ms" stroke="#222220" strokeWidth={2} dot={{ fill: '#222220', r: 4 }} isAnimationActive={false} /></LineChart></ResponsiveContainer></div><div className="chart-footnote">Twelve batch-one runs per length after warm-up on the experiment device. Not a cross-hardware claim.</div></section>
      </div>
      <section className="results-table-section"><div className="section-heading"><div><span>Full result set</span><h2>Held-out model comparison</h2></div></div><div className="results-table"><div className="results-head"><span>Model</span><span>Parameters</span><span>NLL ↓</span><span>Accuracy ↑</span><span>Macro F1 ↑</span><span>Deviation AUROC ↑</span></div>{models.map(model => <div className={`results-row ${model.name === 'Mamba' ? 'highlight' : ''}`} key={model.name}><strong>{model.name}{model.name === 'Mamba' && <span>SUBMITTED</span>}</strong><span className="mono">{model.parameters.toLocaleString()}</span><span className="mono">{model.nll.toFixed(4)}</span><span className="mono">{(model.accuracy * 100).toFixed(2)}%</span><span className="mono">{model.macro_f1.toFixed(4)}</span><span className="mono">{model.synthetic_deviation_auroc?.toFixed(4) ?? 'Not evaluated'}</span></div>)}</div></section>
      <section className="experiment-notes"><div><Check size={18} /><h3>What the result supports</h3><p>Mamba learned useful process structure and produced a higher macro F1 than the evaluated Transformer in this bounded run. Compare NLL, accuracy, and parameter counts above.</p></div><div><Scale size={18} /><h3>What it does not support</h3><p>This is not proof of universal architecture superiority, real fraud detection, production savings, or performance on every procurement process.</p></div><div><FlaskConical size={18} /><h3>Deviation protocol</h3><p>{metrics.models.mamba.note} Paired cases: {metrics.models.mamba.paired_cases}. Deviations include duplicate invoices, premature unblocking, and clearance before receipt.</p></div></section>
    </div>
  )
}
