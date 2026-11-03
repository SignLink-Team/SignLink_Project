interface LiveGlossStripProps {
  /** 서버가 확정한 실시간 단어(글로스) */
  words: string[]
  /** 아직 확정 전인 현재 후보(흐리게 표시) */
  tentative: string | null
  /** 녹화가 끝나 전체 번역을 기다리는 중 */
  isPredicting: boolean
  visible: boolean
}

/**
 * 녹화 중 실시간으로 인식되는 단어를 보여준다.
 * 녹화가 끝나 최종 번역(translation)이 도착하면 App이 words를 비워 이 영역이 사라지고,
 * 전체 문장은 TranslationResult가 보여준다.
 */
export function LiveGlossStrip({ words, tentative, isPredicting, visible }: LiveGlossStripProps) {
  const hasContent = words.length > 0 || Boolean(tentative)

  if (!visible || (!hasContent && !isPredicting)) return null

  return (
    <div className="w-full mt-4 rounded-xl border border-neutral-250 bg-white p-4" aria-live="polite">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-sm font-bold text-neutral-700">
          {isPredicting ? '번역 중...' : '실시간 인식 중'}
        </h3>
        <span className="flex items-center gap-1.5 text-[11px] text-neutral-400">
          <span
            className={`h-2 w-2 animate-pulse rounded-full ${isPredicting ? 'bg-amber-400' : 'bg-red-500'}`}
          />
          {isPredicting ? '녹화 전체를 번역하는 중' : '종료하면 전체 문장으로 바뀝니다'}
        </span>
      </div>

      <div className="flex min-h-8 flex-wrap items-center gap-2">
        {words.map((word, index) => (
          <span
            key={`${word}-${index}`}
            className="rounded-lg bg-neutral-100 px-2.5 py-1 text-sm font-bold text-neutral-800"
          >
            {word}
          </span>
        ))}

        {tentative && (
          <span className="rounded-lg border border-dashed border-neutral-300 px-2.5 py-1 text-sm italic text-neutral-400">
            {tentative}
          </span>
        )}

        {!hasContent && <span className="text-sm text-neutral-400">인식된 단어가 아직 없습니다</span>}
      </div>
    </div>
  )
}