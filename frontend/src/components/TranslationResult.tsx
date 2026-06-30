import { useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Check, CheckCircle2, Copy, History, Languages, RotateCcw, UserCheck } from 'lucide-react'

interface TranslationResultProps {
  text: string | null
  onViewHistory: () => void
  isLoggedIn: boolean
  autoSave: boolean
  onRetry: () => void
}

export const TranslationResult = ({
  text,
  onViewHistory,
  isLoggedIn,
  autoSave,
  onRetry,
}: TranslationResultProps) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    if (!text) return
    navigator.clipboard?.writeText(text)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="w-full mt-6">
      <div className="flex items-center gap-2 mb-3">
        <Languages className="w-5 h-5 text-brand-green" />
        <h2 className="text-lg font-bold font-sans text-on-surface">번역 결과</h2>
      </div>

      <div className="w-full bg-white border border-neutral-100 rounded-2xl p-6 md:p-8 shadow-xs relative min-h-[140px] flex items-center justify-center transition-all duration-300">
        <AnimatePresence mode="wait">
          {!text ? (
            <motion.div
              key="empty-state"
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -5 }}
              className="flex flex-col items-center justify-center text-center p-4"
            >
              <div className="w-10 h-10 rounded-full border border-neutral-200 flex items-center justify-center mb-3">
                <span className="w-4 h-4 rounded-full border border-t-brand-green border-neutral-300 animate-spin" />
              </div>
              <p className="text-base font-semibold font-hyper text-neutral-400">아직 번역된 내용이 없습니다</p>
            </motion.div>
          ) : (
            <motion.div
              key="active-translation"
              initial={{ opacity: 0, scale: 0.99 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.99 }}
              className="w-full flex flex-col"
            >
              <div className="flex items-center justify-between border-b border-neutral-100 pb-3 mb-5">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-full bg-neutral-100 flex items-center justify-center">
                    <UserCheck className="w-4 h-4 text-neutral-500" />
                  </div>
                  <div className="flex items-center gap-1.5 leading-none">
                    <span className="text-sm font-bold text-on-surface">Patient (ASL)</span>
                    <span className="text-xs bg-red-100 text-red-600 px-1.5 py-0.5 rounded-sm font-bold tracking-wider uppercase animate-pulse">
                      LIVE
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-1.5 self-center">
                  <CheckCircle2 className="w-4 h-4 text-brand-green" />
                  <span className="text-xs font-bold text-neutral-500 font-hyper">
                    {autoSave ? 'Synchronized' : 'Local Stream'}
                  </span>
                </div>
              </div>

              <div className="py-2 px-1 text-center font-sans tracking-tight relative">
                <motion.span
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 0.1, scale: 1 }}
                  className="absolute -top-6 left-2 text-7xl font-sans text-neutral-900 pointer-events-none"
                >
                  “
                </motion.span>
                <div className="text-3xl md:text-4xl font-extrabold text-on-surface tracking-tight leading-relaxed max-w-4xl mx-auto dark:text-neutral-900 font-sans break-keep select-all">
                  "{text}"
                </div>
                <motion.span
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 0.1, scale: 1 }}
                  className="absolute -bottom-12 right-2 text-7xl font-sans text-neutral-900 pointer-events-none"
                >
                  ”
                </motion.span>
              </div>

              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mt-8 pt-4 border-t border-neutral-100">
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    onClick={handleCopy}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-semibold text-neutral-500 hover:text-primary hover:bg-neutral-50 rounded-lg transition-all"
                    title="차트에 복사"
                  >
                    {copied ? (
                      <>
                        <Check className="w-4 h-4 text-green-500" />
                        <span className="text-green-500 font-bold">복사 완료</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-4 h-4" />
                        <span>차트에 복사</span>
                      </>
                    )}
                  </button>

                  <button
                    onClick={onRetry}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-semibold text-amber-600 hover:text-amber-700 hover:bg-amber-50 rounded-lg transition-all border border-amber-200/40"
                    title="수어가 잘못 인식된 경우 다시 녹화"
                  >
                    <RotateCcw className="w-4 h-4" />
                    <span>다시 녹화</span>
                  </button>
                </div>

                <p className="text-xs text-neutral-400 font-medium">
                  Atkinson Hyperlegible 고정밀 인식 폰트 적용됨
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {isLoggedIn && (
        <div className="flex justify-center mt-6">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onViewHistory}
            className="flex items-center gap-2 px-5 py-2.5 bg-neutral-100/80 hover:bg-neutral-100 text-on-surface rounded-full text-sm font-bold font-sans transition-all duration-200 border border-neutral-200 shadow-sm cursor-pointer"
          >
            <History className="w-4 h-4 text-neutral-600" />
            <span>기록 보기</span>
          </motion.button>
        </div>
      )}
    </div>
  )
}
