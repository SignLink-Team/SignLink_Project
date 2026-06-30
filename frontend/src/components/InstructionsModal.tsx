import { AnimatePresence, motion } from 'motion/react'
import { ArrowRight, Camera, Hand, HeartPulse, Save, ShieldCheck, X } from 'lucide-react'
import type { ReactNode } from 'react'

interface InstructionsModalProps {
  isOpen: boolean
  onClose: () => void
}

export const InstructionsModal = ({ isOpen, onClose }: InstructionsModalProps) => {
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
                SignLink는 의료진과 수어 사용 환자 간 의사소통을 돕는 실시간 수어 번역 인터페이스입니다.
              </p>

              <div className="space-y-3">
                <GuideStep
                  number="1"
                  icon={<Camera className="w-3.5 h-3.5" />}
                  title="카메라 실행"
                  body="메인 화면의 카메라 영역을 누르면 수어 감지 화면이 켜집니다. 브라우저 카메라 권한이 없으면 합성 추적 화면으로 테스트할 수 있습니다."
                />
                <GuideStep
                  number="2"
                  icon={<Hand className="w-3.5 h-3.5" />}
                  title="임상 수어 시뮬레이터"
                  body="카메라 실행 후 아래 제스처 버튼을 누르면 진료 상황별 번역 결과를 즉시 확인할 수 있습니다."
                />
                <GuideStep
                  number="3"
                  icon={<Save className="w-3.5 h-3.5" />}
                  title="자동 저장"
                  body="의료진 로그인 후 자동 저장을 ON으로 두면 번역 결과가 MongoDB translation_log에 저장됩니다. OFF 상태에서는 화면에만 표시됩니다."
                />
                <GuideStep
                  number="4"
                  icon={<ShieldCheck className="w-3.5 h-3.5" />}
                  title="기록 관리"
                  body="기록 보기에서 저장된 번역을 검색, 필터링, 복사, 삭제할 수 있습니다."
                />
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

function GuideStep({
  number,
  icon,
  title,
  body,
}: {
  number: string
  icon: ReactNode
  title: string
  body: string
}) {
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
