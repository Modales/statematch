export function formatMoney(value: number) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'EUR', maximumFractionDigits: 0 }).format(value)
}

export function formatDate(value: string, withDate = true) {
  const date = new Date(value)
  return new Intl.DateTimeFormat('en-GB', withDate ? { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false } : { hour: '2-digit', minute: '2-digit', hour12: false }).format(date)
}
