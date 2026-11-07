import { chromium } from '@playwright/test'
import { spawn } from 'node:child_process'
import { mkdir, writeFile } from 'node:fs/promises'

const root = new URL('..', import.meta.url).pathname
const output = `${root}/verification`
await mkdir(output, { recursive: true })
const server = spawn(process.execPath, ['scripts/dev.mjs', '--host', '127.0.0.1', '--port', '7100'], { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] })
let serverLog = ''
server.stdout.on('data', chunk => { serverLog += chunk })
server.stderr.on('data', chunk => { serverLog += chunk })
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))
async function waitForServer() {
  for (let attempt = 0; attempt < 90; attempt++) {
    try { const response = await fetch('http://127.0.0.1:7100/api/health'); if (response.ok) return } catch { /* starting */ }
    if (server.exitCode !== null) throw new Error(`Server exited: ${serverLog}`)
    await sleep(300)
  }
  throw new Error(`Server did not start: ${serverLog}`)
}
const report = { checks: [], consoleErrors: [], failedResponses: [], screenshots: [], performance: {} }
const check = (name, value = true) => { report.checks.push({ name, passed: Boolean(value) }); if (!value) throw new Error(`Check failed: ${name}`) }
let browser
try {
  await waitForServer()
  browser = await chromium.launch({ headless: true })
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 })
  const page = await context.newPage()
  page.on('console', message => { if (message.type() === 'error') report.consoleErrors.push(message.text()) })
  page.on('response', response => { if (response.status() >= 400) report.failedResponses.push(`${response.status()} ${response.url()}`) })
  await page.goto('http://127.0.0.1:7100', { waitUntil: 'networkidle' })
  await page.locator('.queue-row').first().waitFor()
  check('Mamba runtime is online', await page.getByText('Mamba online', { exact: true }).isVisible())
  check('Real case queue loaded', await page.locator('.queue-row').count() >= 20)
  await page.screenshot({ path: `${output}/operations-desktop.png`, fullPage: true })
  report.screenshots.push('operations-desktop.png')

  await page.getByRole('button', { name: 'Replay evidence' }).click()
  await page.locator('.evidence-row').first().waitFor()
  check('Counterfactual replay rendered', await page.locator('.evidence-row').count() > 0)
  await page.screenshot({ path: `${output}/evidence-desktop.png`, fullPage: true })
  report.screenshots.push('evidence-desktop.png')

  await page.getByRole('button', { name: /Match ledger/ }).click()
  check('Match ledger rendered', await page.locator('.match-ledger').isVisible())
  await page.getByRole('button', { name: 'Audit chain' }).click()
  await page.locator('.audit-entry').first().waitFor()
  check('Audit chain verified', await page.getByText('Verified', { exact: true }).isVisible())
  await page.screenshot({ path: `${output}/audit-desktop.png`, fullPage: true })
  report.screenshots.push('audit-desktop.png')

  await page.getByRole('button', { name: 'Hold clearance' }).click()
  await page.locator('.toast').waitFor()
  check('Case action persisted', (await page.locator('.toast').textContent()).includes('hold clearance committed'))

  await page.getByRole('button', { name: 'Inject exception' }).click()
  const dialog = page.getByRole('dialog')
  await dialog.getByRole('button').filter({ hasText: 'Duplicate invoice' }).click()
  await page.locator('.case-heading h1').filter({ hasText: 'DEMO-' }).waitFor()
  check('Live exception injection scored', await page.getByText('Repeated invoices within two hours', { exact: true }).isVisible())

  await page.getByRole('button', { name: 'Import CSV' }).click()
  const csv = 'case_id,activity,timestamp,amount,user\nBROWSER-1,Create Purchase Order Item,2026-01-01T09:00:00Z,1250,vendor_17\nBROWSER-1,Record Goods Receipt,2026-01-02T09:00:00Z,1250,user_08\nBROWSER-1,Record Invoice Receipt,2026-01-03T09:00:00Z,1250,user_11\nBROWSER-1,Clear Invoice,2026-01-04T09:00:00Z,1250,user_12\n'
  await page.locator('input[type=file]').setInputFiles({ name: 'browser-trace.csv', mimeType: 'text/csv', buffer: Buffer.from(csv) })
  await page.getByRole('button', { name: 'Import and score' }).click()
  await page.getByText('1 cases imported', { exact: true }).waitFor()
  await page.getByRole('button', { name: 'Close dialog' }).click()
  await page.locator('.case-heading h1').filter({ hasText: 'BROWSER-1' }).waitFor()
  check('CSV upload reached persisted workspace', true)

  await page.getByRole('button', { name: /Model lab/ }).click()
  await page.getByText('Held-out model comparison', { exact: true }).waitFor()
  check('Measured model charts rendered', await page.locator('.recharts-responsive-container').count() === 2)
  await page.locator('.recharts-bar-rectangle').first().waitFor({ state: 'visible' })
  await page.locator('.recharts-line-curve').first().waitFor({ state: 'visible' })
  check('Three measured NLL bars visible', await page.locator('.recharts-bar-rectangle').count() === 3)
  check('Measured latency line visible', await page.locator('.recharts-line-curve').count() === 1)
  await page.screenshot({ path: `${output}/model-lab-desktop.png`, fullPage: true })
  report.screenshots.push('model-lab-desktop.png')

  await page.getByRole('button', { name: /Architecture/ }).click()
  await page.getByText('Selective state update', { exact: true }).waitFor()
  check('Architecture and sources rendered', await page.locator('.research-links a').count() === 3)
  await page.screenshot({ path: `${output}/architecture-desktop.png`, fullPage: true })
  report.screenshots.push('architecture-desktop.png')

  const performance = await page.evaluate(() => {
    const resources = performance.getEntriesByType('resource')
    return { requests: resources.length, transferBytes: resources.reduce((sum, item) => sum + (item.transferSize || 0), 0) }
  })
  report.performance.desktop = performance
  await context.close()

  const mobile = await browser.newContext({ viewport: { width: 375, height: 812 }, deviceScaleFactor: 1, isMobile: true, hasTouch: true })
  const mobilePage = await mobile.newPage()
  mobilePage.on('console', message => { if (message.type() === 'error') report.consoleErrors.push(`mobile: ${message.text()}`) })
  mobilePage.on('response', response => { if (response.status() >= 400) report.failedResponses.push(`mobile: ${response.status()} ${response.url()}`) })
  await mobilePage.goto('http://127.0.0.1:7100', { waitUntil: 'networkidle' })
  await mobilePage.locator('.queue-row').first().waitFor()
  const widths = await mobilePage.evaluate(() => ({ scroll: document.documentElement.scrollWidth, viewport: window.innerWidth }))
  check('No mobile horizontal overflow', widths.scroll <= widths.viewport)
  await mobilePage.screenshot({ path: `${output}/operations-mobile.png`, fullPage: true })
  report.screenshots.push('operations-mobile.png')
  await mobilePage.getByRole('button', { name: 'Open navigation' }).click()
  await mobilePage.getByRole('button', { name: /Model lab/ }).click()
  await mobilePage.getByText('Held-out model comparison', { exact: true }).waitFor()
  check('Mobile navigation works', true)
  await mobile.close()

  check('Zero browser console errors', report.consoleErrors.length === 0)
  check('Zero failed HTTP responses', report.failedResponses.length === 0)
  await writeFile(`${output}/browser-report.json`, JSON.stringify(report, null, 2))
  console.log(JSON.stringify(report, null, 2))
} catch (error) {
  console.error(error)
  console.error(serverLog.slice(-6000))
  await writeFile(`${output}/browser-report.json`, JSON.stringify({ ...report, failure: String(error), serverLog: serverLog.slice(-6000) }, null, 2))
  process.exitCode = 1
} finally {
  if (browser) await browser.close()
  if (server.exitCode === null) {
    server.kill('SIGTERM')
    await Promise.race([new Promise(resolve => server.once('exit', resolve)), sleep(4000)])
    if (server.exitCode === null) server.kill('SIGKILL')
  }
}
