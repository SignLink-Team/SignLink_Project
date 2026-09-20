import { useState, useEffect } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { AlertTriangle, Check, CheckCircle2, Copy, Edit3, History, Languages, RotateCcw, UserCheck, X } from 'lucide-react'
import { TranslationCandidate } from '../types'
import { UnkSolver } from './UnkSolver'
import { useTTS } from '../hooks/useTTS';
import { TTSToggle } from './TTSToggle';

interface TranslationResultProps {
  text: string | null
  glossResult?: string
  candidates?: TranslationCandidate[]
  onViewHistory: () => void
  isLoggedIn: boolean
  autoSave: boolean
  onRetry: () => void
  onSelectTranslation: (text: string, category: string, confidence?: number) => Promise<void>
  onClearTranslation: () => void
}

export const TranslationResult = ({
  text,
  glossResult = '',
  candidates = [],
  onViewHistory,
  isLoggedIn,
  autoSave,
  onRetry,
  onSelectTranslation,
  onClearTranslation,
}: TranslationResultProps) => {
  const [copied, setCopied] = useState(false)
  const [customText, setCustomText] = useState('')

  const hasText = !!text
  const hasUnk = hasText && /<?unk>?/i.test(`${text} ${glossResult}`)

  const handleCopy = () => {
    if (!text) return
    navigator.clipboard?.writeText(text)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 2000)
  }

  const handleApplyCustomText = async () => {
    if (!customText.trim()) return
    await onSelectTranslation(customText.trim(), '수정사항', 1)
    setCustomText('')
  }

  const { speak, stop, isSpeaking } = useTTS();
  const [ttsEnabled, setTtsEnabled] = useState(false);

  useEffect(() => {
    if (ttsEnabled && text) {
      speak(text);
    }
  }, [text, ttsEnabled]);

  const handleToggle = () => {
    setTtsEnabled(prev => {
      const next = !prev;
      if (!next) stop();
      return next;
    });
  };

  useEffect(() => {
    return () => window.speechSynthesis.cancel();
  }, []);

  return (
    <div className="w-full mt-6">
      <div className="flex items-center gap-2 mb-3">
        <Languages className="w-5 h-5 text-brand-green" />
        <h2 className="text-lg font-bold font-sans text-on-surface">번역 결과</h2>
        <TTSToggle enabled={ttsEnabled} onToggle={handleToggle} isSpeaking={isSpeaking} />
      </div>

      <div className="w-full bg-white border border-neutral-100 rounded-2xl p-6 md:p-8 shadow-xs relative min-h-[140px] flex flex-col justify-center transition-all duration-300">
        <AnimatePresence mode="wait">
          {!hasText ? (
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
              <p className="text-base font-semibold text-neutral-400">아직 번역된 내용이 없습니다</p>
            </motion.div>
          ) : hasUnk ? (
            <motion.div
              key="unk-state"
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -5 }}
              className="w-full py-2 flex flex-col"
            >
              <div className="flex flex-col items-center justify-center text-center max-w-xl mx-auto mb-6">
                <div className="w-12 h-12 rounded-full bg-amber-50 border border-amber-200 flex items-center justify-center mb-3 text-amber-500 shadow-sm">
                  <AlertTriangle className="w-6 h-6" />
                </div>
                <h3 className="text-lg font-bold text-on-surface">실시간 수집 불가 또는 UNK 감지됨</h3>
                <p className="text-sm text-neutral-500 mt-2 font-medium leading-relaxed">
                  인식 결과에 UNK가 포함되었습니다. AI 서버가 제공한 후보 문장을 선택하거나 직접 올바른 문장을 입력해 저장하세요.
                </p>
              </div>

              <div className="w-full bg-neutral-50 rounded-xl p-4 border border-neutral-200 mb-5 text-left">
                <div className="flex items-center gap-1.5 mb-2.5">
                  <Edit3 className="w-4 h-4 text-secondary" />
                  <span className="text-xs font-bold text-neutral-600 uppercase tracking-widest">문장 직접 작성</span>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={customText}
                    onChange={(event) => setCustomText(event.target.value)}
                    className="flex-1 px-4 py-2.5 bg-white border border-neutral-250 focus:border-secondary focus:ring-1 focus:ring-secondary rounded-lg text-sm outline-none font-medium text-on-surface shadow-3xs"
                    placeholder="저장할 환자 문장을 직접 작성해 주세요"
                  />
                  <button
                    onClick={handleApplyCustomText}
                    className="px-4 py-2.5 bg-secondary hover:bg-secondary-dark text-white rounded-lg text-sm font-bold cursor-pointer transition-all active:scale-95 shadow-sm whitespace-nowrap"
                  >
                    문장 적용
                  </button>
                </div>
              </div>

              <UnkSolver
                glossResult={glossResult || text}
                candidates={candidates}
                onSolveComplete={(resolvedText, confidence) => {
                  onSelectTranslation(resolvedText, 'AI 복원', confidence)
                  setCustomText('')
                }}
              />
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
                    <span className="text-sm font-bold text-on-surface">Patient (KSL)</span>
                    <span className="text-xs bg-red-100 text-red-600 px-1.5 py-0.5 rounded-sm font-bold tracking-wider uppercase animate-pulse">
                      LIVE
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-1.5 self-center">
                  <CheckCircle2 className="w-4 h-4 text-brand-green" />
                  <span className="text-xs font-bold text-neutral-500">{autoSave ? 'Synchronized' : 'Local Stream'}</span>
                </div>
              </div>

              <div className="py-2 px-1 text-center font-sans tracking-tight relative">
                <div className="text-3xl md:text-4xl font-extrabold text-on-surface tracking-tight leading-relaxed max-w-4xl mx-auto dark:text-neutral-900 font-sans break-keep select-all">
                  "{text}"
                </div>
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
                    title="다시 인식"
                  >
                    <RotateCcw className="w-4 h-4" />
                    <span>다시 인식</span>
                  </button>

                  <button
                    onClick={onClearTranslation}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-semibold text-neutral-500 hover:text-neutral-700 hover:bg-neutral-50 rounded-lg transition-all"
                  >
                    <X className="w-4 h-4" />
                    초기화
                  </button>
                </div>

                <p className="text-xs text-neutral-400 font-medium">Atkinson Hyperlegible 고정밀 인식 폰트 적용됨</p>
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
