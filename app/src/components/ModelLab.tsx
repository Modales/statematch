import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Check, Clock3, FlaskConical, Gauge, Scale } from 'lucide-react'
import type { Metrics } from '@/types'

const deviationNames: Record<string, string> = {
  duplicate_invoice: 'Duplicate',
  premature_unblock: 'Unblock',
  clear_before_gr: 'Early clear',
}

function interval(values?: number[], percent = false) {
  if (!values) return 'Not available'
  const format = (value: number) => percent ? `${(value * 100).toFixed(2)}%` : value.toFixed(3)
  return `${format(values[0])}–${format(values[1])}`
}

export function ModelLab({ metrics }: { metrics: Metrics }) {
  const models = [
    { name: 'Mamba', ...metrics.models.mamba },
    { name: 'Transformer', ...metrics.models.transformer },
    { name: 'Markov', ...metrics.models.markov },
  ]
  const latency = metrics.models.mamba.latency || []
  const extended = metrics.extended_evaluation
  const driftData = extended ? [
    { name: 'Anchor', nll: extended.anchor_800.nll },
    ...extended.later_windows.map((window, index) => ({ name: `Later ${index + 1}`, nll: window.nll })),
  ] : []
  const deviationData = extended ? Object.entries(extended.deviations.by_kind).map(([kind, result]) => ({
    name: deviationNames[kind] ?? kind,
    auroc: result.auroc,
    pairs: result.pairs,
  })) : []

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

      {extended && <>
        <section className="extended-header" data-testid="extended-evaluation">
          <div className="section-heading"><div><span>ACADEMIC EXTENSION / FROZEN CHECKPOINT</span><h2>Later-period validation</h2><p>The original 800-case anchor is replicated, then every remaining chronological test case is scored without retraining or threshold tuning.</p></div><Clock3 size={22} /></div>
          <div className="extended-metrics">
            <div><span>Later cases</span><strong>{extended.later_period.cases.toLocaleString()}</strong><small>{extended.full_temporal_test.cases.toLocaleString()} in the complete temporal test</small></div>
            <div><span>Later NLL ↓</span><strong>{extended.later_period.nll.toFixed(4)}</strong><small>95% CI {interval(extended.later_period.nll_ci95)}</small></div>
            <div><span>Later accuracy ↑</span><strong>{(extended.later_period.accuracy * 100).toFixed(2)}%</strong><small>95% CI {interval(extended.later_period.accuracy_ci95, true)}</small></div>
            <div><span>Deviation AUROC ↑</span><strong>{extended.deviations.auroc.toFixed(4)}</strong><small>95% CI {interval(extended.deviations.auroc_ci95)}</small></div>
          </div>
          <div className="extended-method">{extended.bootstrap.repetitions} whole-case bootstrap resamples · checkpoint <span>{extended.checkpoint_sha256.slice(0, 12)}</span> · no test-case selection</div>
        </section>
        <div className="lab-grid">
          <section className="chart-section"><div className="section-heading"><div><span>Case-end ordered evaluation</span><h2>Temporal drift is visible</h2></div><Clock3 size={19} /></div><div className="chart-frame" aria-label="Later-period temporal drift chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={driftData} margin={{ top: 12, right: 24, left: 0, bottom: 0 }}><CartesianGrid vertical={false} stroke="#deded8" /><XAxis dataKey="name" tickLine={false} axisLine={false} padding={{ left: 24, right: 24 }} /><YAxis domain={[0.6, 1.4]} ticks={[0.6, 0.8, 1, 1.2, 1.4]} tickFormatter={(value) => Number(value).toFixed(1)} width={42} tickLine={false} axisLine={false} /><Tooltip contentStyle={{ border: '1px solid #c8c8c1', borderRadius: 2, background: '#f9f9f6' }} formatter={(value) => Number(value).toFixed(4)} /><Line type="monotone" dataKey="nll" name="NLL" stroke="#a3452c" strokeWidth={2} dot={{ fill: '#a3452c', r: 4 }} isAnimationActive={false} /></LineChart></ResponsiveContainer></div><div className="chart-footnote">The anchor contains the original 800 test cases. Three equal-count windows cover the remaining {extended.later_period.cases.toLocaleString()} cases, ordered by case end.</div></section>
          <section className="chart-section"><div className="section-heading"><div><span>{extended.deviations.pairs.toLocaleString()} independent source cases</span><h2>Deviation separation by type</h2></div><Scale size={19} /></div><div className="chart-frame" aria-label="Deviation AUROC by injection type"><ResponsiveContainer width="100%" height="100%"><BarChart data={deviationData} margin={{ top: 12, right: 12, left: -16, bottom: 0 }}><CartesianGrid vertical={false} stroke="#deded8" /><XAxis dataKey="name" tickLine={false} axisLine={false} interval={0} tick={{ fontSize: 9 }} /><YAxis domain={[0, 1]} tickLine={false} axisLine={false} /><Tooltip cursor={{ fill: '#ecece7' }} contentStyle={{ border: '1px solid #c8c8c1', borderRadius: 2, background: '#f9f9f6' }} formatter={(value, name) => name === 'auroc' ? Number(value).toFixed(4) : Number(value).toLocaleString()} /><Bar dataKey="auroc" name="auroc" fill="#222220" barSize={42} isAnimationActive={false} /></BarChart></ResponsiveContainer></div><div className="chart-footnote">At most one deterministic injection per source case. {(extended.deviations.score_increase_rate * 100).toFixed(2)}% of mutated traces score higher than their paired original. This is not observed fraud detection.</div></section>
        </div>
        <section className="research-basis"><span>Research basis</span><p>Chronological evaluation follows the predictive-monitoring benchmark tradition; anomaly results are reported by injected type with explicit limits.</p><div><a href="https://dl.acm.org/doi/abs/10.1145/3301300" target="_blank" rel="noreferrer">Teinemaa et al. · 2019</a><a href="https://link.springer.com/article/10.1007/s12599-023-00794-y" target="_blank" rel="noreferrer">Ko &amp; Comuzzi · 2023</a><a href="https://arxiv.org/abs/1902.03155" target="_blank" rel="noreferrer">Nolle et al. · 2019</a></div></section>
      </>}

      <section className="experiment-notes"><div><Check size={18} /><h3>What the result supports</h3><p>Mamba learned useful process structure and produced a higher macro F1 than the evaluated Transformer in this bounded run. Compare NLL, accuracy, and parameter counts above.</p></div><div><Scale size={18} /><h3>What it does not support</h3><p>This is not proof of universal architecture superiority, real fraud detection, production savings, or stable future performance. The later-period degradation remains visible above.</p></div><div><FlaskConical size={18} /><h3>Deviation protocol</h3><p>{metrics.models.mamba.note} Paired cases: {metrics.models.mamba.paired_cases}. Deviations include duplicate invoices, premature unblocking, and clearance before receipt.</p></div></section>
    </div>
  )
}
