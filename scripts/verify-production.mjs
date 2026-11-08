import { chromium } from '@playwright/test'
import { spawn } from 'node:child_process'
import { mkdir, rm, writeFile } from 'node:fs/promises'

const root = new URL('..', import.meta.url).pathname
const output = `${root}/verification`
const database = `${output}/production.sqlite3`
const port = 7101
const base = `http://127.0.0.1:${port}`
await mkdir(output, { recursive: true })
await rm(database, { force: true })
const server = spawn('python3', ['-m', 'uvicorn', 'app.main:app', '--app-dir', 'api', '--host', '127.0.0.1', '--port', String(port)], {
  cwd: root,
  env: { ...process.env, STATEMATCH_DB: database },
  stdio: ['ignore', 'pipe', 'pipe'],
})
let serverLog = ''
server.stdout.on('data', chunk => { serverLog += chunk })
server.stderr.on('data', chunk => { serverLog += chunk })
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))
const report = { checks: [], consoleErrors: [], failedResponses: [] }
const check = (name, value = true) => {
  report.checks.push({ name, passed: Boolean(value) })
  if (!value) throw new Error(`Check failed: ${name}`)
}
let browser
try {
  let healthy = false
  for (let attempt = 0; attempt < 90; attempt++) {
    try { healthy = (await fetch(`${base}/api/health`)).ok } catch { /* starting */ }
    if (healthy) break
    if (server.exitCode !== null) throw new Error(`Server exited: ${serverLog}`)
    await sleep(300)
  }
  check('Production API cold start', healthy)
  browser = await chromium.launch({ headless: true })
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
  page.on('console', message => { if (message.type() === 'error') report.consoleErrors.push(message.text()) })
  page.on('response', response => { if (response.status() >= 400) report.failedResponses.push(`${response.status()} ${response.url()}`) })
  await page.goto(base, { waitUntil: 'networkidle' })
  await page.locator('.queue-row').first().waitFor()
  check('FastAPI serves compiled application', await page.locator('.app-shell').isVisible())
  check('Real checkpoint online in production mode', await page.getByText('Mamba online', { exact: true }).isVisible())
  check('Seeded production queue', await page.locator('.queue-row').count() === 27)
  await page.getByRole('button', { name: /Model lab/ }).click()
  await page.locator('.recharts-bar-rectangle').first().waitFor({ state: 'visible' })
  await page.locator('.recharts-line-curve').first().waitFor({ state: 'visible' })
  check('Lazy model lab renders measured marks', await page.locator('.chart-frame[aria-label="Model NLL comparison"] .recharts-bar-rectangle').count() === 3)
  check('Production extended evaluation renders', await page.getByTestId('extended-evaluation').isVisible() && await page.locator('.chart-frame[aria-label="Later-period temporal drift chart"] .recharts-line-curve').count() === 1)
  await page.screenshot({ path: `${output}/production-model-lab.png`, fullPage: true })
  const fallback = await fetch(`${base}/client/route`)
  check('SPA fallback remains available', fallback.ok && (await fallback.text()).includes('<div id="root"></div>'))
  const health = await (await fetch(`${base}/api/health`)).json()
  check('Production health reports no LLM or attention', health.model_loaded && !health.llm && !health.attention)
  check('Zero production console errors', report.consoleErrors.length === 0)
  check('Zero production HTTP failures', report.failedResponses.length === 0)
  await writeFile(`${output}/production-report.json`, JSON.stringify(report, null, 2))
  console.log(JSON.stringify(report, null, 2))
} catch (error) {
  console.error(error)
  console.error(serverLog.slice(-6000))
  await writeFile(`${output}/production-report.json`, JSON.stringify({ ...report, failure: String(error), serverLog: serverLog.slice(-6000) }, null, 2))
  process.exitCode = 1
} finally {
  if (browser) await browser.close()
  if (server.exitCode === null) {
    server.kill('SIGTERM')
    await Promise.race([new Promise(resolve => server.once('exit', resolve)), sleep(4000)])
    if (server.exitCode === null) server.kill('SIGKILL')
  }
  await rm(database, { force: true })
}
