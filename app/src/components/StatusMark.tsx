import type { CaseStatus } from '@/types'

const labels: Record<CaseStatus, string> = {
  exception: 'Exception', review: 'Review', ready: 'Ready', open: 'Open', reconciled: 'Reconciled', held: 'Held', in_review: 'In review', awaiting_receipt: 'Awaiting receipt',
}

export function StatusMark({ status }: { status: CaseStatus }) {
  return <span className={`status-mark status-${status}`}><span aria-hidden="true" />{labels[status]}</span>
}

