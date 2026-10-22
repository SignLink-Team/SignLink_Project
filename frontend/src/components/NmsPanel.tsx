import { NMS_KEYS, NMS_KEY_COLORS, NMS_KEY_THRESHOLD, NmsFrame, NmsSegment } from '../nms'

interface NmsPanelProps {
  frame: NmsFrame | null
  segments: NmsSegment[]
  visible: boolean
}

export function NmsPanel({ frame, segments, visible }: NmsPanelProps) {
  if (!visible && !frame && segments.length === 0) return null

  return (
    <div className="w-full mt-4 rounded-xl border border-neutral-250 bg-white p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-bold text-neutral-700">비수지 신호</h3>
        <span className="text-[11px] text-neutral-400">
          {frame ? `frame ${frame.frameId}` : '대기 중'}
        </span>
      </div>

      <div className="flex flex-col gap-1.5">
        {NMS_KEYS.map((key, index) => {
          const value = frame?.smoothed[index] ?? 0
          const active = frame?.active[index] ?? false
          const color = NMS_KEY_COLORS[key]
          const threshold = NMS_KEY_THRESHOLD[key]

          return (
            <div key={key} className="flex items-center gap-2">
              <span
                className="w-12 shrink-0 text-xs font-bold"
                style={{ color: active ? color : '#9ca3af' }}
              >
                {key}
              </span>

              <div className="relative h-3 flex-1 overflow-hidden rounded bg-neutral-200">
                <div
                  className="h-full rounded transition-[width] duration-100"
                  style={{
                    width: `${Math.min(1, Math.max(0, value)) * 100}%`,
                    backgroundColor: active ? color : '#9ca3af',
                  }}
                />
                {/* 활성 임계값 표시 */}
                <div
                  className="absolute top-0 h-full w-px bg-neutral-600/60"
                  style={{ left: `${threshold * 100}%` }}
                />
              </div>

              <span className="w-10 shrink-0 text-right text-[11px] tabular-nums text-neutral-500">
                {value.toFixed(2)}
              </span>
              <span
                className="h-2 w-2 shrink-0 rounded-full"
                style={{ backgroundColor: active ? color : 'transparent' }}
              />
            </div>
          )
        })}
      </div>

      {segments.length > 0 && (
        <div className="mt-4 border-t border-neutral-200 pt-3">
          <h4 className="mb-1.5 text-xs font-bold text-neutral-600">비수지 타임라인</h4>
          <ul className="flex flex-col gap-0.5 text-[11px] text-neutral-500">
            {segments.map((seg, i) => (
              <li key={`${seg.key}-${seg.start}-${i}`} className="tabular-nums">
                <span className="inline-block w-12 font-bold">{seg.key}</span>
                {seg.start.toFixed(2)}s ~ {seg.end.toFixed(2)}s (지속 {(seg.end - seg.start).toFixed(2)}s)
                {seg.peak !== undefined && ` · 강도 ${seg.peak.toFixed(2)}`}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}