import { FormEvent, useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { ArrowLeft, Check, Clock, Copy, FileSpreadsheet, Filter, Plus, Search, Trash2 } from 'lucide-react'
import { TranslationLog } from '../types'

interface HistoryViewProps {
  logs: TranslationLog[]
  onBack: () => void
  onClearLogs: () => Promise<void>
  onDeleteLog: (id: string) => Promise<void>
  onAddManualLog: (text: string, category: string) => Promise<void>
}

const fallbackCategories = ['전체', '두통', '호흡기', '전신/감기', '소화기', '근골격계', '알레르기', '이비인후과', '기타']

export const HistoryView = ({
  logs,
  onBack,
  onClearLogs,
  onDeleteLog,
  onAddManualLog,
}: HistoryViewProps) => {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('전체')
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [manualText, setManualText] = useState('')
  const [manualCat, setManualCat] = useState('기타')
  const [showAddForm, setShowAddForm] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const categories = useMemo(() => {
    const list = new Set(fallbackCategories)
    logs.forEach((log) => {
      if (log.category) list.add(log.category)
    })
    return Array.from(list)
  }, [logs])

  const filteredLogs = useMemo(() => (
    logs.filter((log) => {
      const matchSearch = log.text.toLowerCase().includes(searchTerm.toLowerCase())
      const matchCat = selectedCategory === '전체' || log.category === selectedCategory
      return matchSearch && matchCat
    })
  ), [logs, searchTerm, selectedCategory])

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard?.writeText(text)
    setCopiedId(id)
    window.setTimeout(() => setCopiedId(null), 1500)
  }

  const handleManualAddSubmit = async (event: FormEvent) => {
    event.preventDefault()
    if (!manualText.trim()) return
    setError(null)
    try {
      await onAddManualLog(manualText.trim(), manualCat)
      setManualText('')
      setShowAddForm(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : '기록 추가에 실패했습니다.')
    }
  }

  const handleDelete = async (id: string) => {
    setError(null)
    try {
      await onDeleteLog(id)
    } catch (err) {
      setError(err instanceof Error ? err.message : '기록 삭제에 실패했습니다.')
    }
  }

  const handleClear = async () => {
    if (!window.confirm('전체 번역 기록을 삭제할까요?')) return
    setError(null)
    try {
      await onClearLogs()
    } catch (err) {
      setError(err instanceof Error ? err.message : '전체 삭제에 실패했습니다.')
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className="max-w-4xl mx-auto px-4 py-6 font-sans"
    >
      <div className="flex items-center justify-between border-b border-neutral-100 pb-4 mb-4">
        <div className="flex items-center gap-3">
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={onBack}
            className="p-2 hover:bg-neutral-100 rounded-full transition-colors duration-150"
            title="대시보드로 돌아가기"
          >
            <ArrowLeft className="w-5 h-5 text-on-surface" />
          </motion.button>

          <div className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-primary" />
            <h2 className="text-xl font-extrabold text-on-surface">번역 기록</h2>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowAddForm((value) => !value)}
            className="flex items-center gap-1.5 px-3.5 py-2 bg-secondary hover:bg-secondary-dark text-white rounded-lg text-sm font-bold transition-all shadow-xs"
          >
            <Plus className="w-4 h-4" />
            <span>기록 직접 추가</span>
          </button>

          {logs.length > 0 && (
            <button
              onClick={handleClear}
              className="flex items-center gap-1.5 px-3.5 py-2 border border-red-200 hover:bg-red-50 text-red-600 rounded-lg text-sm font-bold transition-all"
            >
              <Trash2 className="w-4 h-4" />
              <span>전체 비우기</span>
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm font-bold text-red-700">
          {error}
        </div>
      )}

      <AnimatePresence>
        {showAddForm && (
          <motion.form
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            onSubmit={handleManualAddSubmit}
            className="mb-5 bg-secondary-container/20 border border-secondary-container/40 rounded-xl p-4 overflow-hidden"
          >
            <h3 className="text-sm font-bold text-secondary mb-3">수동 진료 기록 작성</h3>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-3 items-end">
              <div className="md:col-span-2">
                <label className="block text-xs font-bold text-neutral-600 mb-1.5">수어 번역 텍스트</label>
                <input
                  type="text"
                  required
                  value={manualText}
                  onChange={(event) => setManualText(event.target.value)}
                  placeholder="예: 오른쪽 팔에 통증이 있습니다."
                  className="w-full px-3 py-2 bg-white border border-neutral-200 rounded-lg text-sm focus:border-secondary outline-none"
                />
              </div>
              <div>
                <label className="block text-xs font-bold text-neutral-600 mb-1.5">진료 분류</label>
                <select
                  value={manualCat}
                  onChange={(event) => setManualCat(event.target.value)}
                  className="w-full px-3 py-2 bg-white border border-neutral-200 rounded-lg text-sm focus:border-secondary outline-none"
                >
                  {fallbackCategories.filter((cat) => cat !== '전체').map((cat) => (
                    <option key={cat} value={cat}>{cat}</option>
                  ))}
                </select>
              </div>
              <div className="flex gap-2">
                <button type="submit" className="w-full px-4 py-2 bg-secondary text-white rounded-lg text-sm font-bold hover:bg-secondary-dark">
                  저장
                </button>
                <button
                  type="button"
                  onClick={() => setShowAddForm(false)}
                  className="px-3 py-2 bg-neutral-200 text-on-surface rounded-lg text-sm font-bold hover:bg-neutral-300"
                >
                  취소
                </button>
              </div>
            </div>
          </motion.form>
        )}
      </AnimatePresence>

      <div className="w-full bg-white border border-neutral-100 rounded-xl p-4 shadow-xs mb-4">
        <div className="relative">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-neutral-400" />
          <input
            type="text"
            value={searchTerm}
            onChange={(event) => setSearchTerm(event.target.value)}
            placeholder="번역 내용 검색..."
            className="w-full pl-9 pr-4 py-2 bg-neutral-50 hover:bg-neutral-100/60 focus:bg-white border border-neutral-200 focus:border-primary rounded-lg text-sm font-medium outline-none transition-all duration-200 font-hyper text-on-surface"
          />
        </div>

        <div className="flex flex-wrap gap-1.5 mt-3 items-center">
          <span className="text-xs font-extrabold text-neutral-400 mr-1.5 flex items-center gap-0.5 uppercase tracking-wide">
            <Filter className="w-2.5 h-2.5" />
            분류 필터:
          </span>
          {categories.map((cat) => {
            const isSelected = selectedCategory === cat
            return (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-3 py-1 rounded-full text-xs sm:text-sm font-bold leading-none border transition-all duration-200 ${
                  isSelected
                    ? 'bg-primary text-white border-primary shadow-xs'
                    : 'bg-neutral-50 hover:bg-neutral-100 text-neutral-500 border-neutral-200'
                }`}
              >
                {cat}
              </button>
            )
          })}
        </div>
      </div>

      <div className="space-y-2.5">
        <AnimatePresence initial={false}>
          {filteredLogs.length > 0 ? (
            filteredLogs.map((log, index) => (
              <motion.div
                key={log.id}
                layoutId={log.id}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.98 }}
                transition={{ duration: 0.18, delay: Math.min(index * 0.03, 0.2) }}
                className="w-full bg-white hover:bg-neutral-50/50 border border-neutral-100 rounded-xl p-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 transition-all duration-150 shadow-2xs"
              >
                <div className="flex items-start gap-3.5 flex-1 min-w-0">
                  <div className="w-8 h-8 rounded-lg bg-neutral-100 flex items-center justify-center flex-shrink-0 border border-neutral-150">
                    <span className="text-secondary font-bold text-sm select-none">...</span>
                  </div>

                  <div className="space-y-1.5 pr-2">
                    <p className="text-base font-semibold text-on-surface font-sans leading-snug break-keep select-all">
                      {log.text}
                    </p>
                    <div className="flex items-center gap-2">
                      {log.category && (
                        <span className="text-xxs sm:text-xs font-bold bg-secondary-container/15 text-secondary border border-secondary-container/30 px-2 py-0.5 rounded-sm tracking-wide">
                          {log.category}
                        </span>
                      )}
                      {log.isCustom && (
                        <span className="text-xxs sm:text-xs font-bold bg-neutral-100 text-neutral-500 border border-neutral-200 px-1.5 py-0.5 rounded-sm uppercase">
                          Custom
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between sm:justify-end gap-4 border-t sm:border-t-0 pt-2 sm:pt-0 border-neutral-50">
                  <div className="flex items-center gap-1.5 text-right font-hyper">
                    <Clock className="w-3.5 h-3.5 text-neutral-300" />
                    <span className="text-xs sm:text-sm text-neutral-400 font-bold whitespace-nowrap">{log.timestamp}</span>
                  </div>

                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => handleCopy(log.id, log.text)}
                      className="p-1.5 hover:bg-neutral-100 rounded-md text-neutral-400 hover:text-primary transition-colors duration-100"
                      title="클립보드에 복사"
                    >
                      {copiedId === log.id ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                    </button>
                    <button
                      onClick={() => handleDelete(log.id)}
                      className="p-1.5 hover:bg-red-50 rounded-md text-neutral-400 hover:text-red-600 transition-colors duration-100"
                      title="항목 삭제"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </motion.div>
            ))
          ) : (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="py-12 bg-white/50 border border-dashed border-neutral-200 rounded-xl flex flex-col items-center justify-center text-center"
            >
              <FileSpreadsheet className="w-10 h-10 text-neutral-300 mb-2.5" />
              <p className="text-sm font-bold text-neutral-400 font-hyper">조건에 맞는 번역 기록이 없습니다.</p>
              <p className="text-xs text-neutral-400 font-medium mt-1.5 font-sans">다른 검색어를 사용하거나 새 번역을 생성하세요.</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}
