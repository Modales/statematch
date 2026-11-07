import { spawn } from 'node:child_process'

const args = process.argv.slice(2)
const python = process.env.PYTHON || 'python3'
const api = spawn(python, ['-m', 'uvicorn', 'app.main:app', '--app-dir', 'api', '--host', '127.0.0.1', '--port', '8000'], {
  stdio: 'inherit', env: process.env,
})
const vite = spawn(process.execPath, ['app/node_modules/vite/bin/vite.js', 'app', ...args], {
  stdio: 'inherit', env: process.env,
})
let closing = false
function close(code = 0) {
  if (closing) return
  closing = true
  api.kill('SIGTERM')
  vite.kill('SIGTERM')
  setTimeout(() => process.exit(code), 300).unref()
}
api.on('exit', (code) => { if (!closing && code) close(code) })
vite.on('exit', (code) => { if (!closing) close(code || 0) })
process.on('SIGINT', () => close(0))
process.on('SIGTERM', () => close(0))
