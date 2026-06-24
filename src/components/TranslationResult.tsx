/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Languages, 
  History, 
  CheckCircle2, 
  UserCheck, 
  Copy, 
  Check, 
  RotateCcw, 
  Edit3, 
  AlertTriangle, 
  RefreshCw, 
  CornerDownRight 
} from 'lucide-react';
import { GESTURE_PRESETS } from '../data';
import { UnkSolver } from './UnkSolver';

interface TranslationResultProps {
  text: string | null;
  onViewHistory: () => void;
  isLoggedIn: boolean;
  autoSave: boolean;
  onSelectTranslation: (text: string, category: string) => void;
  onClearTranslation: () => void;
}

export const TranslationResult: React.FC<TranslationResultProps> = ({
  text,
  onViewHistory,
  isLoggedIn,
  autoSave,
  onSelectTranslation,
  onClearTranslation,
}) => {
  const [copied, setCopied] = useState(false);
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [customText, setCustomText] = useState('');
  const [isEditing, setIsEditing] = useState(false);

  // Checks if the text represents an empty or unk state
  const isUnk = !text || text.toLowerCase().trim() === 'unk';

  // Copy to clipboard
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
    setIsEditing(false);
  };

  const handleStartEditing = () => {
    setCustomText(text && text.toLowerCase().trim() !== 'unk' ? text : '');
    setIsEditing(true);
  };

  return (
    <div className="w-full mt-6">
      {/* Title Header Block */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Languages className="w-5 h-5 text-brand-green" />
          <h2 className="text-lg font-bold font-sans text-on-surface">
            번역 결과
          </h2>
        </div>
        {text && !isUnk && (
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => {
              setIsPanelOpen(!isPanelOpen);
              setIsEditing(false);
            }}
            className="flex items-center gap-1.5 px-3 py-1 bg-secondary-container/30 hover:bg-secondary-container/50 text-secondary border border-secondary-container/40 rounded-full text-xs font-bold transition-all cursor-pointer animate-pulse"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>재번역 / 보정</span>
          </motion.button>
        )}
      </div>

      {/* Main Translation Viewer Box */}
      <div className="w-full bg-white border border-neutral-100 rounded-2xl p-6 md:p-8 shadow-xs relative min-h-[140px] flex flex-col justify-center transition-all duration-300">
        <AnimatePresence mode="wait">
          {isUnk ? (
            /* Empty or unk (Unknown) State */
            <motion.div
              key="unk-state"
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -5 }}
              className="w-full py-2 flex flex-col"
            >
              <div className="flex flex-col items-center justify-center text-center max-w-xl mx-auto mb-6">
                {text ? (
                  <div className="w-12 h-12 rounded-full bg-amber-50 border border-amber-200 flex items-center justify-center mb-3 text-amber-500 animate-pulse shadow-sm">
                    <AlertTriangle className="w-6 h-6" />
                  </div>
                ) : (
                  <div className="w-10 h-10 rounded-full border border-neutral-200 flex items-center justify-center mb-3">
                    <span className="w-4 h-4 rounded-full border border-t-brand-green border-neutral-300 animate-spin" />
                  </div>
                )}
                
                <h3 className="text-lg font-bold font-sans text-on-surface">
                  {text ? '⚠️ 실시간 수집 불가 (unk 감지됨)' : '아직 번역된 내용이 없습니다'}
                </h3>
                <p className="text-sm text-neutral-500 mt-2 font-medium leading-relaxed">
                  {text 
                    ? '수어 동작이 불명확하거나 부분적으로 추적되어 결과가 "unk"로 출력되었습니다. 아래 unk (수어 단어/문장 사전)에서 알맞은 수어를 선택하여 덮어쓰거나, 직접 올바른 소견을 입력하여 보정해 주세요.'
                    : '스마트 카메라가 수어를 인식하여 실시간 번역 결과를 렌더링합니다. 현재 상태에서 즉시 소견을 대입하려면 아래 unk (수어 단어/문장 사전) 목록에서 선택해 주세요.'
                  }
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
                    placeholder={text ? 'unk를 대체하여 저장할 환자 소견을 직접 작성해 주세요' : '즉시 차트에 대입할 환자 임상 소견을 작성하세요'}
                  />
                  <button
                    onClick={handleApplyCustomText}
                    className="px-4 py-2.5 bg-secondary hover:bg-secondary-dark text-white rounded-lg text-sm font-bold cursor-pointer transition-all active:scale-95 shadow-sm whitespace-nowrap"
                  >
                    소견 적용
                  </button>
                </div>
              </div>

              {/* unk (수어 단어/문장 사전) Grid */}
              <UnkSolver
                onSolveComplete={(result) => {
                onSelectTranslation(result, 'AI 복원');
                setCustomText('');
              }}
              />


              {/* Quick feedback layer when the user might think it's wrong */}
              <div className="mt-3 py-2 px-3 bg-neutral-50 border border-neutral-150 rounded-lg text-xs font-medium text-neutral-500 flex items-center justify-between sm:max-w-md mx-auto">
                <span>💡 실시간 번역 결과가 의료진 의도와 다르게 나왔나요?</span>
                <button
                  onClick={() => {
                    setIsPanelOpen(true);
                    setIsEditing(false);
                    // smooth scroll
                    const el = document.getElementById('assist-panel');
                    el?.scrollIntoView({ behavior: 'smooth' });
                  }}
                  className="text-secondary font-bold hover:underline cursor-pointer ml-2 whitespace-nowrap"
                >
                  재번역 / 보정 열기
                </button>
              </div>

              {/* Utility row inside box */}
              <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 mt-8 pt-4 border-t border-neutral-100">
                <div className="flex items-center gap-2">
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

                  <button
                    onClick={() => {
                      setIsPanelOpen(!isPanelOpen);
                      setIsEditing(false);
                    }}
                    className={`flex items-center gap-1.5 px-3.5 py-1.5 text-sm font-bold rounded-lg transition-all border cursor-pointer ${
                      isPanelOpen 
                        ? 'bg-secondary/10 border-secondary/30 text-secondary' 
                        : 'bg-neutral-50 hover:bg-neutral-100 text-neutral-600 border-neutral-200 shadow-3xs'
                    }`}
                  >
                    <RotateCcw className="w-4 h-4" />
                    <span>재번역 / 수정</span>
                  </button>
                </div>

                <p className="text-xs text-neutral-400 font-medium self-center">
                  Atkinson Hyperlegible 고정밀 인식 폰트 적용됨
                </p>
              </div>

              {/* Interactive Re-translate / Overwrite Control Panel */}
              <AnimatePresence>
                {isPanelOpen && (
                  <motion.div
                    id="assist-panel"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-5 border-t border-dashed border-neutral-200 pt-5 overflow-hidden"
                  >
                    <div className="bg-neutral-50 rounded-xl p-4 md:p-5 border border-neutral-200 text-left animate-slide-up">
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-1.5">
                          <AlertTriangle className="w-4 h-4 text-secondary" />
                          <h4 className="text-sm font-bold text-neutral-800">
                            의료 수어 번역 보정 & 실시간 재분석 레이어
                          </h4>
                        </div>
                        <button 
                          onClick={() => {
                            setIsPanelOpen(false);
                            setIsEditing(false);
                          }}
                          className="text-xs font-bold text-neutral-455 hover:text-neutral-600 cursor-pointer"
                        >
                          접기
                        </button>
                      </div>

                      <div className="flex flex-wrap gap-2.5 mb-4">
                        <button
                          onClick={() => {
                            onClearTranslation();
                            setIsPanelOpen(false);
                          }}
                          className="px-3.5 py-1.5 bg-white border border-neutral-250 hover:border-neutral-350 hover:bg-neutral-50 text-xs font-bold text-neutral-700 rounded-lg flex items-center gap-1 transition-all cursor-pointer"
                        >
                          <RefreshCw className="w-3.5 h-3.5 text-neutral-500" />
                          <span>초기화 후 재감지</span>
                        </button>

                        <button
                          onClick={handleStartEditing}
                          className="px-3.5 py-1.5 bg-white border border-neutral-250 hover:border-neutral-350 hover:bg-neutral-50 text-xs font-bold text-neutral-700 rounded-lg flex items-center gap-1 transition-all cursor-pointer"
                        >
                          <Edit3 className="w-3.5 h-3.5 text-neutral-500" />
                          <span>소견 직접 편집하기</span>
                        </button>
                      </div>

                      {/* Manual text edit layout */}
                      {isEditing && (
                        <div className="space-y-2.5 mb-4 p-3 bg-white border border-neutral-200 rounded-lg">
                          <label className="block text-[11px] font-bold text-neutral-500 uppercase">
                            환자 소견 문구 직접 덮어쓰기
                          </label>
                          <div className="flex items-center gap-2">
                            <input
                              type="text"
                              value={customText}
                              onChange={(e) => setCustomText(e.target.value)}
                              className="flex-1 px-3 py-1.5 bg-neutral-50 border border-neutral-200 rounded-md text-xs focus:ring-1 focus:ring-secondary focus:border-secondary focus:bg-white outline-none"
                              placeholder="번역된 텍스트 수정하기"
                            />
                            <button
                              onClick={handleApplyCustomText}
                              className="px-3 py-1.5 bg-secondary hover:bg-secondary-dark text-white rounded-md text-xs font-bold cursor-pointer"
                            >
                              적용
                            </button>
                          </div>
                        </div>
                      )}

                      {/* Overwrite list directory */}
                      <div>
                        <p className="text-[11px] font-bold text-neutral-550 uppercase mb-2">
                          unk (수어 단어/문장 사전)에서 올바른 수어 골라 덮어쓰기:
                        </p>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                          {GESTURE_PRESETS.map((preset) => (
                            <button
                              key={preset.id}
                              onClick={() => {
                                onSelectTranslation(preset.translationText, preset.category);
                                setIsPanelOpen(false);
                              }}
                              className={`p-2.5 rounded-lg border text-left transition-all hover:bg-white cursor-pointer group ${
                                text === preset.translationText 
                                  ? 'border-secondary bg-secondary-container/15 text-secondary font-bold' 
                                  : 'border-neutral-200 text-neutral-600 bg-white/50 hover:border-neutral-300'
                              }`}
                            >
                              <div className="flex items-center justify-between mb-0.5">
                                <span className="text-[9px] font-bold bg-neutral-100 text-neutral-500 px-1 rounded truncate max-w-full">
                                  {preset.category}
                                </span>
                              </div>
                              <p className="text-xs font-semibold truncate">
                                {preset.gestureName.split(' (')[0]}
                              </p>
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          ) : (
            /* Active Live Translation Frame */
            <motion.div
              key="active-translation"
              initial={{ opacity: 0, scale: 0.99 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.99 }}
              className="w-full flex flex-col"
            >
              {/* Context bar inside Translation Box */}
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

                {/* Status Toggles on right */}
                <div className="flex items-center gap-1.5 self-center">
                  <CheckCircle2 className="w-4 h-4 text-brand-green" />
                  <span className="text-xs font-bold text-neutral-550 font-hyper">
                    {autoSave ? 'Synchronized' : 'Local Stream'}
                  </span>
                </div>
              </div>

              {/* Highlegibility Double Quote Translation Panel */}
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

              {/* Quick feedback layer when the user might think it's wrong */}
              <div className="mt-3 py-2 px-3 bg-neutral-50 border border-neutral-150 rounded-lg text-xs font-medium text-neutral-500 flex items-center justify-between sm:max-w-md mx-auto">
                <span>💡 실시간 번역 결과가 의료진 의도와 다르게 나왔나요?</span>
                <button
                  type="button"
                  onClick={() => {
                    setIsPanelOpen(true);
                    setIsEditing(false);
                    const el = document.getElementById('assist-panel');
                    el?.scrollIntoView({ behavior: 'smooth' });
                  }}
                  className="text-secondary font-bold hover:underline cursor-pointer ml-2 whitespace-nowrap"
                >
                  재번역 / 보정 열기
                </button>
              </div>

              {/* Utility row inside box */}
              <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 mt-8 pt-4 border-t border-neutral-100">
                <div className="flex items-center gap-2">
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

                  <button
                    onClick={() => {
                      setIsPanelOpen(!isPanelOpen);
                      setIsEditing(false);
                    }}
                    className={`flex items-center gap-1.5 px-3.5 py-1.5 text-sm font-bold rounded-lg transition-all border cursor-pointer ${
                      isPanelOpen 
                        ? 'bg-secondary/10 border-secondary/30 text-secondary' 
                        : 'bg-neutral-50 hover:bg-neutral-100 text-neutral-600 border-neutral-200 shadow-3xs'
                    }`}
                  >
                    <RotateCcw className="w-4 h-4" />
                    <span>재번역 / 수정</span>
                  </button>
                </div>

                <p className="text-xs text-neutral-400 font-medium self-center">
                  Atkinson Hyperlegible 고정밀 인식 폰트 적용됨
                </p>
              </div>

              {/* Interactive Re-translate / Overwrite Control Panel */}
              <AnimatePresence>
                {isPanelOpen && (
                  <motion.div
                    id="assist-panel"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-5 border-t border-dashed border-neutral-200 pt-5 overflow-hidden"
                  >
                    <div className="bg-neutral-50 rounded-xl p-4 md:p-5 border border-neutral-200 text-left animate-slide-up">
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-1.5">
                          <AlertTriangle className="w-4 h-4 text-secondary" />
                          <h4 className="text-sm font-bold text-neutral-800">
                            의료 수어 번역 보정 & 실시간 재분석 레이어
                          </h4>
                        </div>
                        <button 
                          onClick={() => {
                            setIsPanelOpen(false);
                            setIsEditing(false);
                          }}
                          className="text-xs font-bold text-neutral-455 hover:text-neutral-600 cursor-pointer"
                        >
                          접기
                        </button>
                      </div>

                      <div className="flex flex-wrap gap-2.5 mb-4">
                        <button
                          onClick={() => {
                            onClearTranslation();
                            setIsPanelOpen(false);
                          }}
                          className="px-3.5 py-1.5 bg-white border border-neutral-250 hover:border-neutral-350 hover:bg-neutral-50 text-xs font-bold text-neutral-700 rounded-lg flex items-center gap-1 transition-all cursor-pointer"
                        >
                          <RefreshCw className="w-3.5 h-3.5 text-neutral-500" />
                          <span>초기화 후 재감지</span>
                        </button>

                        <button
                          onClick={handleStartEditing}
                          className="px-3.5 py-1.5 bg-white border border-neutral-250 hover:border-neutral-350 hover:bg-neutral-50 text-xs font-bold text-neutral-700 rounded-lg flex items-center gap-1 transition-all cursor-pointer"
                        >
                          <Edit3 className="w-3.5 h-3.5 text-neutral-500" />
                          <span>소견 직접 편집하기</span>
                        </button>
                      </div>

                      {/* Manual text edit layout */}
                      {isEditing && (
                        <div className="space-y-2.5 mb-4 p-3 bg-white border border-neutral-200 rounded-lg">
                          <label className="block text-[11px] font-bold text-neutral-500 uppercase">
                            환자 소견 문구 직접 덮어쓰기
                          </label>
                          <div className="flex items-center gap-2">
                            <input
                              type="text"
                              value={customText}
                              onChange={(e) => setCustomText(e.target.value)}
                              className="flex-1 px-3 py-1.5 bg-neutral-50 border border-neutral-200 rounded-md text-xs focus:ring-1 focus:ring-secondary focus:border-secondary focus:bg-white outline-none"
                              placeholder="번역된 텍스트 수정하기"
                            />
                            <button
                              onClick={handleApplyCustomText}
                              className="px-3 py-1.5 bg-secondary hover:bg-secondary-dark text-white rounded-md text-xs font-bold cursor-pointer"
                            >
                              적용
                            </button>
                          </div>
                        </div>
                      )}

                      {/* Overwrite list directory */}
                      <div>
                        <p className="text-[11px] font-bold text-neutral-550 uppercase mb-2">
                          unk (수어 단어/문장 사전)에서 올바른 수어 골라 덮어쓰기:
                        </p>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                          {GESTURE_PRESETS.map((preset) => (
                            <button
                              key={preset.id}
                              onClick={() => {
                                onSelectTranslation(preset.translationText, preset.category);
                                setIsPanelOpen(false);
                              }}
                              className={`p-2.5 rounded-lg border text-left transition-all hover:bg-white cursor-pointer group ${
                                text === preset.translationText 
                                  ? 'border-secondary bg-secondary-container/15 text-secondary font-bold' 
                                  : 'border-neutral-200 text-neutral-600 bg-white/50 hover:border-neutral-300'
                              }`}
                            >
                              <div className="flex items-center justify-between mb-0.5">
                                <span className="text-[9px] font-bold bg-neutral-100 text-neutral-500 px-1 rounded truncate max-w-full">
                                  {preset.category}
                                </span>
                              </div>
                              <p className="text-xs font-semibold truncate">
                                {preset.gestureName.split(' (')[0]}
                              </p>
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
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
