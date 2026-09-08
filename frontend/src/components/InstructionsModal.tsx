import { AnimatePresence, motion } from 'motion/react'
import { ArrowRight, Camera, Hand, HeartPulse, Save, ShieldCheck, X, Pencil } from 'lucide-react'
import type { ReactNode } from 'react'

interface InstructionsModalProps {
  isOpen: boolean
  onClose: () => void
  isLoggedIn?: boolean
}

export const InstructionsModal = ({ isOpen, onClose, isLoggedIn }: InstructionsModalProps) => {
  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.6 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-neutral-900"
          />

          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 15 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 15 }}
            className="bg-white rounded-2xl border border-neutral-100 max-w-lg w-full p-6 shadow-2xl relative z-10 overflow-hidden font-sans"
          >
            <button
              onClick={onClose}
              className="absolute right-4 top-4 p-1.5 rounded-full hover:bg-neutral-100 text-neutral-400 hover:text-neutral-700 transition-colors"
              title="닫기"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-2.5 border-b border-neutral-100 pb-4 pr-6">
              <div className="w-9 h-9 rounded-full bg-secondary/10 flex items-center justify-center text-secondary">
                <HeartPulse className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-on-surface">SignLink 사용 가이드</h3>
                <p className="text-[11px] text-neutral-400 font-medium">실시간 의료 수어 번역 서비스</p>
              </div>
            </div>

            <div className="mt-4 space-y-4">
              <p className="text-xs text-neutral-500 leading-relaxed font-medium">
                SignLink는 의료진과 수어 사용자 사이의 의사소통을 돕는 실시간 수어 번역 인터페이스입니다.
              </p>

              <div className="space-y-3">
                <GuideStep
                  number="1"
                  icon={<Camera className="w-3.5 h-3.5" />}
                  title="카메라 맞추기"
                  body="환자의 상반신과 손동작이 화면 중앙에 잘 보이도록 자리에 위치해주세요."
                />
                <GuideStep
                  number="2"
                  icon={<Hand className="w-3.5 h-3.5" />}
                  title="AI 예측"
                  body="AI가 사용자의 수어 동작을 분석하여 문장으로 번역한 결과를 화면에 실시간으로 표시합니다."
                />
                <GuideStep
                  number="3"
                  icon={<Save className="w-3.5 h-3.5" />}
                  title="번역 내용 저장"
                  body="의료진으로 로그인한 경우, 자동 저장을 켜두면 번역된 내용을 진료 기록으로 저장할 수 있습니다."
                />
                {isLoggedIn && (
                <>
                <GuideStep
                  number="4"
                  icon={<ShieldCheck className="w-3.5 h-3.5" />}
                  title="기록 확인 및 관리"
                  body="저장된 번역 기록을 언제든 다시 조회·복사하여 진료에 활용하거나 삭제할 수 있습니다."
                />
                <GuideStep
                  number="5"
                  icon={<Pencil className="w-3.5 h-3.5" />}
                  title="환자 ID 등록"
                  body="저장된 번역 기록에 환자 ID를 등록할 수 있습니다."
                />
                </>
                )}
              </div>
            </div>

            <div className="mt-6 pt-4 border-t border-neutral-100 flex justify-end">
              <button
                onClick={onClose}
                className="flex items-center gap-1 px-4 py-2 bg-secondary text-white rounded-lg text-xs font-bold hover:bg-secondary-dark transition-colors"
              >
                <span>확인했습니다</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  )
}

function GuideStep({ number, icon, title, body }: { number: string; icon: ReactNode; title: string; body: string }) {
  return (
    <div className="flex gap-3 items-start">
      <div className="w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold leading-none shrink-0 mt-0.5">
        {number}
      </div>
      <div>
        <h4 className="text-xs font-bold text-on-surface flex items-center gap-1">
          {icon}
          {title}
        </h4>
        <p className="text-[11px] text-neutral-500 mt-0.5 leading-relaxed">{body}</p>
      </div>
    </div>
  )
}