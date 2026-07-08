import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import {
  Languages,
  History,
  CheckCircle2,
  UserCheck,
  Copy,
  Check,
  Edit3,
  AlertTriangle,
} from 'lucide-react';
import { UnkSolver } from './UnkSolver';
import { useTTS } from '../hooks/useTTS';
import { TTSToggle } from './TTSToggle';

interface TranslationResultProps {
  text: string | null;
  onViewHistory: () => void;
  isLoggedIn: boolean;
  autoSave: boolean;
  onSelectTranslation: (text: string, category: string) => void;
  onClearTranslation: () => void;
  onRetry?: () => void;
}

export const TranslationResult: React.FC<TranslationResultProps> = ({
  text,
  onViewHistory,
  isLoggedIn,
  autoSave,
  onSelectTranslation,
  onClearTranslation,
  onRetry,
}) => {
  const [copied, setCopied] = useState(false);
  const [customText, setCustomText] = useState('');

  const hasText = !!text;
  const hasUnk = hasText && /unk/i.test(text as string);

  const handleCopy = () => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleApplyCustomText = () => {
    if (!customText.trim()) return;
    onSelectTranslation(customText.trim(), '수정사항');
    setCustomText('');
  };

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
      {/* Title Header Block */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Languages className="w-5 h-5 text-brand-green" />
          <h2 className="text-lg font-bold font-sans text-on-surface">번역 결과</h2>
          <TTSToggle enabled={ttsEnabled} onToggle={handleToggle} isSpeaking={isSpeaking} />
        </div>
      </div>

      {/* Main Translation Viewer Box */}
      <div className="w-full bg-white border border-neutral-100 rounded-2xl p-6 md:p-8 shadow-xs relative min-h-[140px] flex flex-col justify-center transition-all duration-300">
        <AnimatePresence mode="wait">
          {!hasText ? (
            /* 1) 아직 번역 시작 전 - UnkSolver 안 보임 */
            <motion.div
              key="idle-state"
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -5 }}
              className="w-full py-2 flex flex-col items-center text-center"
            >
              <div className="w-10 h-10 rounded-full border border-neutral-200 flex items-center justify-center mb-3">
                <span className="w-4 h-4 rounded-full border border-t-brand-green border-neutral-300 animate-spin" />
              </div>
              <h3 className="text-lg font-bold font-sans text-on-surface">
                아직 번역된 내용이 없습니다
              </h3>
              <p className="text-sm text-neutral-500 mt-2 font-medium leading-relaxed max-w-xl">
                스마트 카메라가 수어를 인식하면 실시간 번역 결과가 여기에 표시됩니다.
              </p>
            </motion.div>
          ) : hasUnk ? (
            /* 2) 번역 결과에 unk가 포함된 경우 - UnkSolver 표시 */
            <motion.div
              key="unk-state"
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -5 }}
              className="w-full py-2 flex flex-col"
            >
              <div className="flex flex-col items-center justify-center text-center max-w-xl mx-auto mb-6">
                <div className="w-12 h-12 rounded-full bg-amber-50 border border-amber-200 flex items-center justify-center mb-3 text-amber-500 animate-pulse shadow-sm">
                  <AlertTriangle className="w-6 h-6" />
                </div>
                <h3 className="text-lg font-bold font-sans text-on-surface">
                  ⚠️ 실시간 수집 불가 (unk 감지됨)
                </h3>
                <p className="text-sm text-neutral-500 mt-2 font-medium leading-relaxed">
                  수어 동작이 불명확하거나 부분적으로 추적되어 결과에 "unk"가 포함되었습니다. 아래 AI 문맥 복원 엔진에서 추천하는 문장을 선택하거나, 직접 올바른 소견을 입력하여 보정해 주세요.
                </p>
              </div>

              {/* Quick Text Input for correcting/creating translation text */}
              <div className="w-full bg-neutral-50 rounded-xl p-4 border border-neutral-200 mb-5 text-left">
                <div className="flex items-center gap-1.5 mb-2.5">
                  <Edit3 className="w-4 h-4 text-secondary" />
                  <span className="text-xs font-bold text-neutral-600 uppercase tracking-widest font-hyper">
                    소견 직접 작성 및 덮어쓰기
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={customText}
                    onChange={(e) => setCustomText(e.target.value)}
                    className="flex-1 px-4 py-2.5 bg-white border border-neutral-250 focus:border-secondary focus:ring-1 focus:ring-secondary rounded-lg text-sm outline-none font-medium text-on-surface shadow-3xs"
                    placeholder="unk를 대체하여 저장할 환자 소견을 직접 작성해 주세요"
                  />
                  <button
                    onClick={handleApplyCustomText}
                    className="px-4 py-2.5 bg-secondary hover:bg-secondary-dark text-white rounded-lg text-sm font-bold cursor-pointer transition-all active:scale-95 shadow-sm whitespace-nowrap"
                  >
                    소견 적용
                  </button>
                </div>
              </div>

              {/* AI 문맥 복원 엔진 (UnkSolver) */}
              <UnkSolver
                onSolveComplete={(result) => {
                  onSelectTranslation(result, 'AI 복원');
                  setCustomText('');
                }}
              />

              <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 mt-8 pt-4 border-t border-neutral-100">
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1.5 px-3.5 py-1.5 bg-neutral-50 hover:bg-neutral-100 text-sm font-bold text-neutral-600 rounded-lg transition-all border border-neutral-200 cursor-pointer shadow-3xs"
                  title="Copy text to clipboard"
                >
                  {copied ? (
                    <>
                      <Check className="w-4 h-4 text-green-500 animate-scale" />
                      <span className="text-green-500 font-bold">복사 완료!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-4 h-4" />
                      <span>차트에 복사</span>
                    </>
                  )}
                </button>

                <p className="text-xs text-neutral-400 font-medium self-center">
                  Atkinson Hyperlegible 고정밀 인식 폰트 적용됨
                </p>
              </div>
            </motion.div>
          ) : (
            /* 3) 정상적으로 번역된 경우 */
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
                  <div>
                    <div className="flex items-center gap-1.5 leading-none">
                      <span className="text-sm font-bold text-on-surface">Patient (ASL)</span>
                      <span className="text-xs bg-red-100 text-red-600 px-1.5 py-0.5 rounded-sm font-bold tracking-wider uppercase animate-pulse">
                        LIVE
                      </span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-1.5 self-center">
                  <CheckCircle2 className="w-4 h-4 text-brand-green" />
                  <span className="text-xs font-bold text-neutral-550 font-hyper">
                    {autoSave ? 'Synchronized' : 'Local Stream'}
                  </span>
                </div>
              </div>

              <div className="py-2 px-1 text-center font-sans tracking-tight relative my-3">
                <motion.span
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 0.1, scale: 1 }}
                  className="absolute -top-6 left-2 text-7xl font-sans text-neutral-900 pointer-events-none"
                >
                  “
                </motion.span>
                <div className="text-3xl md:text-4xl font-extrabold text-on-surface tracking-tight leading-relaxed max-w-4xl mx-auto dark:text-neutral-900 font-sans break-keep select-all">
                  {text}
                </div>
                <motion.span
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 0.1, scale: 1 }}
                  className="absolute -bottom-12 right-2 text-7xl font-sans text-neutral-900 pointer-events-none"
                >
                  ”
                </motion.span>
              </div>

              <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 mt-8 pt-4 border-t border-neutral-100">
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1.5 px-3.5 py-1.5 bg-neutral-50 hover:bg-neutral-100 text-sm font-bold text-neutral-600 rounded-lg transition-all border border-neutral-200 cursor-pointer shadow-3xs"
                  title="Copy text to clipboard"
                >
                  {copied ? (
                    <>
                      <Check className="w-4 h-4 text-green-500 animate-scale" />
                      <span className="text-green-500 font-bold">복사 완료!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-4 h-4" />
                      <span>차트에 복사</span>
                    </>
                  )}
                </button>

                <p className="text-xs text-neutral-400 font-medium self-center">
                  Atkinson Hyperlegible 고정밀 인식 폰트 적용됨
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* View logs callout */}
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
  );
};