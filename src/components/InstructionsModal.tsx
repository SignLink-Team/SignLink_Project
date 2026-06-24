/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { X, Hand, Camera, Save, ArrowRight, ShieldCheck, HeartPulse } from 'lucide-react';

interface InstructionsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const InstructionsModal: React.FC<InstructionsModalProps> = ({ isOpen, onClose }) => {
  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop screen */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.6 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-neutral-900"
          />

          {/* Modal Container */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 15 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 15 }}
            className="bg-white rounded-2xl border border-neutral-100 max-w-lg w-full p-6 shadow-2xl relative z-10 overflow-hidden font-sans"
          >
            {/* Close Button */}
            <button
              onClick={onClose}
              className="absolute right-4 top-4 p-1.5 rounded-full hover:bg-neutral-100 text-neutral-400 hover:text-neutral-700 transition-colors"
              title="Close modal"
            >
              <X className="w-5 h-5" />
            </button>

            {/* Header */}
            <div className="flex items-center gap-2.5 border-b border-neutral-100 pb-4 pr-6">
              <div className="w-9 h-9 rounded-full bg-secondary/10 flex items-center justify-center text-secondary">
                <HeartPulse className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-on-surface">SignLink 사용 설명 가이드</h3>
                <p className="text-[11px] text-neutral-400 font-medium">실시간 의료 수어 번역 서비스</p>
              </div>
            </div>

            {/* Body instruction steps */}
            <div className="mt-4 space-y-4">
              <p className="text-xs text-neutral-500 leading-relaxed font-medium">
                의료진과 농인/난청 환자 간의 원활한 소통을 지원하는 AI 수어 번역 플랫폼 <strong>SignLink</strong>의 핵심 기능을 테스트해보세요:
              </p>

              <div className="space-y-3">
                {/* Step 1 */}
                <div className="flex gap-3 items-start">
                  <div className="w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold leading-none shrink-0 mt-0.5">
                    1
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-on-surface flex items-center gap-1">
                      <Camera className="w-3.5 h-3.5" />
                      스마트 수어 감지 실행
                    </h4>
                    <p className="text-[11px] text-neutral-500 mt-0.5 leading-relaxed">
                      대시보드 중앙의 <strong>진한 초록색 유어포트 카드</strong>를 클리하면 실시간 추적이 개시됩니다. 실제 카메라 허용 시 실화면 노출과 더불어, 가상 골격 트래킹 궤적 알고리즘이 중첩 구동됩니다.
                    </p>
                  </div>
                </div>

                {/* Step 2 */}
                <div className="flex gap-3 items-start">
                  <div className="w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold leading-none shrink-0 mt-0.5">
                    2
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-on-surface flex items-center gap-1">
                      <Hand className="w-3.5 h-3.5" />
                      의료 맞춤 수어 제스처 시뮬레이션
                    </h4>
                    <p className="text-[11px] text-neutral-500 mt-0.5 leading-relaxed">
                      카메라 작동 시 표출되는 하단 <strong>시뮬레이터 제스처 칩</strong>(두통, 기침, 복통 등)을 눌러 기등록 수어의 자연스러운 문장 변역 결과를 즉각 피드백 받으세요.
                    </p>
                  </div>
                </div>

                {/* Step 3 */}
                <div className="flex gap-3 items-start">
                  <div className="w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold leading-none shrink-0 mt-0.5">
                    3
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-on-surface flex items-center gap-1">
                      <Save className="w-3.5 h-3.5" />
                      자동 저장 및 전자 의무 기록(EMR) 연동
                    </h4>
                    <p className="text-[11px] text-neutral-500 mt-0.5 leading-relaxed">
                      로그인 완료 시 헤더에서 <strong>자동 저장 ON</strong> 상태가 점등되며, 이후 생성되는 모든 진료 수어 구문은 자동으로 전자의무기록(EMR) 대장에 자동 동기화 처리됩니다.
                    </p>
                  </div>
                </div>

                {/* Step 4 */}
                <div className="flex gap-3 items-start">
                  <div className="w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold leading-none shrink-0 mt-0.5">
                    4
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-on-surface flex items-center gap-1">
                      <ShieldCheck className="w-3.5 h-3.5" />
                      테스트용 로그인 요령
                    </h4>
                    <p className="text-[11px] text-neutral-500 mt-0.5 leading-relaxed">
                      메인 우측 '로그인' 선택 후, 입력창의 고정 이메일(<code>name@hospital.com</code>) 규격으로 자유롭게 즉각 로그인을 처리할 수 있습니다.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* CTA action footer */}
            <div className="mt-6 pt-4 border-t border-neutral-100 flex justify-end">
              <button
                onClick={onClose}
                className="flex items-center gap-1 px-4 py-2 bg-secondary text-white rounded-lg text-xs font-bold hover:bg-secondary-dark transition-colors"
              >
                <span>이해했습니다</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>

          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};
