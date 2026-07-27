import { useState } from 'react'
import { motion } from 'motion/react'
import { Check, ChevronRight, Cpu, Sparkles } from 'lucide-react'
import { TranslationCandidate } from '../types'

interface UnkSolverProps {
  glossResult: string
  candidates: TranslationCandidate[]
  onSolveComplete: (resolvedText: string, confidence: number) => void
}

export const UnkSolver = ({ glossResult, candidates, onSolveComplete }: UnkSolverProps) => {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null)

  const glosses = glossResult.split(/\s+/).filter(Boolean)
  const unkCount = glosses.filter((gloss) => gloss.toUpperCase() === 'UNK' || gloss.toUpperCase() === '<UNK>').length

  const handleConfirm = () => {
    if (selectedIndex === null) return
    const selected = candidates[selectedIndex]
    onSolveComplete(selected.text, selected.confidence)
  }

  return (
    <div className="w-full bg-white border border-neutral-150 rounded-2xl p-5 md:p-6 shadow-sm">
      <div className="flex items-center gap-2 mb-5 border-b border-neutral-100 pb-4">
        <Cpu className="w-5 h-5 text-brand-green" />
        <div>
          <h3 className="font-bold text-sm text-on-surface">AI 문맥 복원 엔진</h3>
          <p className="text-[11px] text-neutral-400 font-medium">
            AI 서버가 전달한 후보 문장 중 진료 상황에 가장 적절한 문장을 선택하세요.
          </p>
        </div>
      </div>

      <div className="bg-neutral-50 border border-neutral-150 rounded-xl p-4 mb-4">
        <div className="flex items-center justify-between mb-2.5">
          <p className="text-xs font-bold text-neutral-550 uppercase tracking-wider">실시간 수어 인식 글로스</p>
          {unkCount > 0 && (
            <span className="text-[10px] bg-red-50 text-red-500 font-bold px-2 py-0.5 rounded-md">
              UNK {unkCount}개 감지됨
            </span>
          )}
        </div>

        <div className="flex flex-wrap gap-2">
          {glosses.map((gloss, index) => {
            const isUnk = gloss.toUpperCase() === 'UNK' || gloss.toUpperCase() === '<UNK>'
            return (
              <span
                key={`${gloss}-${index}`}
                className={`px-3 py-1.5 rounded-lg border text-xs font-bold ${
                  isUnk ? 'bg-red-50 text-red-500 border-red-200' : 'bg-white border-neutral-200 text-neutral-700'
                }`}
              >
                {isUnk ? 'UNK' : gloss}
              </span>
            )
          })}
        </div>
      </div>

      {candidates.length === 0 ? (
        <div className="rounded-xl border border-dashed border-neutral-200 bg-neutral-50 p-5 text-center">
          <Sparkles className="w-6 h-6 text-neutral-300 mx-auto mb-2" />
          <p className="text-sm font-bold text-neutral-500">AI 서버 후보 문장이 아직 없습니다.</p>
          <p className="text-xs text-neutral-400 mt-1">
            이후 AI 서버 LLM이 translation_candidates 필드로 후보 문장을 내려주면 이 영역에 표시됩니다.
          </p>
        </div>
      ) : (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-2.5">
          <p className="text-xs font-bold text-neutral-550 mb-1">추천 문장 후보</p>

          {candidates.map((candidate, index) => {
            const selected = selectedIndex === index
            return (
              <button
                key={`${candidate.text}-${index}`}
                type="button"
                onClick={() => setSelectedIndex(index)}
                className={`w-full text-left p-3.5 rounded-xl border transition-all flex items-center justify-between gap-3 cursor-pointer ${
                  selected
                    ? 'bg-brand-green/10 border-brand-green ring-1 ring-brand-green shadow-xs'
                    : 'bg-white border-neutral-200 hover:border-neutral-300 hover:bg-neutral-50'
                }`}
              >
                <div className="flex items-center gap-3 min-w-0">
                  <div
                    className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 text-[10px] font-bold ${
                      selected ? 'bg-brand-green text-white' : 'bg-neutral-100 text-neutral-500'
                    }`}
                  >
                    {selected ? <Check className="w-3.5 h-3.5" /> : index + 1}
                  </div>
                  <span className="font-bold text-sm text-on-surface truncate">{candidate.text}</span>
                </div>
                <span
                  className={`text-[10px] font-bold px-2 py-0.5 rounded-md shrink-0 ${
                    selected ? 'bg-brand-green text-white' : 'bg-neutral-100 text-neutral-500'
                  }`}
                >
                  {Math.round(candidate.confidence)}%
                </span>
              </button>
            )
          })}

          <button
            type="button"
            onClick={handleConfirm}
            disabled={selectedIndex === null}
            className="w-full flex items-center justify-center gap-1.5 py-2.5 bg-brand-green hover:bg-emerald-600 disabled:bg-neutral-100 disabled:text-neutral-400 text-white rounded-xl font-bold text-sm transition-all cursor-pointer shadow-sm"
          >
            <ChevronRight className="w-4 h-4" />
            이 문장으로 확정
          </button>
        </motion.div>
      )}
    </div>
  )
}
