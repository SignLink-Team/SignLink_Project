import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Sparkles, Cpu, Check, RefreshCw, ChevronRight } from 'lucide-react';

interface UnkSolverProps {
  onSolveComplete: (resolvedText: string) => void;
}

interface SentenceCandidate {
  text: string;
  confidence: number;
}

interface SelectedPreset {
  id: string;
  glosses: string[];
  description: string;
  sentenceCandidates: SentenceCandidate[];
}

const SOLVER_PRESETS: SelectedPreset[] = [
  {
    id: 'breast-cancer',
    glosses: ['유방암', 'UNK', 'UNK', '있다'],
    description: '유방암(암) 유무 및 병력 청취',
    sentenceCandidates: [
      { text: '유방암 진단을 받은 적 있다', confidence: 96 },
      { text: '유방암 수술을 받은 적 있다', confidence: 81 },
      { text: '유방암 검사를 받은 적 있다', confidence: 74 },
    ],
  },
  {
    id: 'fever-cold',
    glosses: ['감기1', 'UNK', '시작1'],
    description: '감기나 고열 증상의 최초 발현 시점 유추',
    sentenceCandidates: [
      { text: '감기 어제부터 시작', confidence: 94 },
      { text: '감기 오늘부터 시작', confidence: 79 },
      { text: '감기 며칠 전부터 시작', confidence: 68 },
    ],
  },
  {
    id: 'chest-pain',
    glosses: ['가슴1', 'UNK', '아프다2'],
    description: '호흡기/심장 통증의 디테일한 양상 유추',
    sentenceCandidates: [
      { text: '가슴 콕콕찌르듯 아프다', confidence: 93 },
      { text: '가슴 답답하게 아프다', confidence: 77 },
      { text: '가슴 뻐근하게 아프다', confidence: 70 },
    ],
  },
];

export const UnkSolver: React.FC<UnkSolverProps> = ({ onSolveComplete }) => {
  const [activePreset, setActivePreset] = useState<SelectedPreset>(SOLVER_PRESETS[0]);
  const [isSolving, setIsSolving] = useState<boolean>(false);
  const [candidates, setCandidates] = useState<SentenceCandidate[] | null>(null);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [confirmed, setConfirmed] = useState<boolean>(false);

  const handleSelectPreset = (preset: SelectedPreset) => {
    setActivePreset(preset);
    setCandidates(null);
    setSelectedIndex(null);
    setConfirmed(false);
  };

  const executeUnkSolve = () => {
    if (isSolving) return;
    setIsSolving(true);
    setCandidates(null);
    setSelectedIndex(null);
    setConfirmed(false);

    // TODO: 실제 연동 시 여기서 Claude API 등에 글로스 시퀀스를 보내고
    // 응답으로 상위 N개 문장 후보 + confidence를 받아오도록 교체
    setTimeout(() => {
      setCandidates(activePreset.sentenceCandidates);
      setIsSolving(false);
    }, 1200);
  };

  const handleConfirm = () => {
    if (selectedIndex === null || !candidates) return;
    const chosen = candidates[selectedIndex].text;
    setConfirmed(true);
    onSolveComplete(chosen);
  };

  const unkCount = activePreset.glosses.filter((g) => g.toUpperCase() === 'UNK').length;

  return (
    <div className="w-full bg-white border border-neutral-150 rounded-2xl p-5 md:p-6 shadow-sm">
      {/* Header & Presets */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-5 border-b border-neutral-100 pb-4">
        <div className="flex items-center gap-2">
          <Cpu className="w-5 h-5 text-brand-green animate-pulse" />
          <div>
            <h3 className="font-bold text-sm text-on-surface">AI 문맥 복원 엔진</h3>
            <p className="text-[10px] text-neutral-400 font-medium">
              인식 불가(UNK) 요소가 포함된 문장을 AI가 예측한 후보 중에서 선택하세요.
            </p>
          </div>
        </div>

        <div className="flex flex-wrap gap-1.5">
          {SOLVER_PRESETS.map((p) => (
            <button
              key={p.id}
              type="button"
              onClick={() => handleSelectPreset(p)}
              className={`px-3 py-1 rounded-full text-[10px] font-bold transition-all border cursor-pointer ${
                activePreset.id === p.id
                  ? 'bg-brand-green/10 text-brand-green border-brand-green'
                  : 'bg-neutral-50 hover:bg-neutral-100/70 text-neutral-500 border-neutral-200'
              }`}
            >
              {p.description.split(' ')[0] || p.id}
            </button>
          ))}
        </div>
      </div>

      {/* Gloss sequence (read-only) */}
      <div className="bg-neutral-50 border border-neutral-150 rounded-xl p-4 mb-4">
        <div className="flex items-center justify-between mb-2.5">
          <p className="text-xs font-bold text-neutral-550 uppercase tracking-wider">
            실시간 수어 인식 글로스
          </p>
          {unkCount > 0 && (
            <span className="text-[10px] bg-red-50 text-red-500 font-bold px-2 py-0.5 rounded-md">
              UNK {unkCount}개 감지됨
            </span>
          )}
        </div>

        <div className="flex flex-wrap gap-2">
          {activePreset.glosses.map((g, idx) => {
            const isUnk = g.toUpperCase() === 'UNK';
            return (
              <span
                key={idx}
                className={`px-3 py-1.5 rounded-lg border text-xs font-bold shadow-3xs ${
                  isUnk
                    ? 'bg-red-50 text-red-500 border-red-250'
                    : 'bg-white border-neutral-200 text-neutral-700'
                }`}
              >
                {isUnk ? 'UNK' : g}
              </span>
            );
          })}
        </div>

        <p className="text-[11px] text-neutral-400 mt-3 font-medium">
          💡 AI가 전체 문맥을 분석해 가장 가능성 높은 문장 후보 3개를 제안합니다.
        </p>
      </div>

      {/* Solve trigger */}
      {!candidates && (
        <button
          type="button"
          onClick={executeUnkSolve}
          disabled={isSolving}
          className="
            w-full py-3
            bg-brand-green hover:bg-emerald-600
            disabled:bg-neutral-100 disabled:text-neutral-400
            text-white rounded-xl
            font-bold text-sm cursor-pointer transition-all shadow-sm flex items-center justify-center gap-1.5
          "
        >
          {isSolving ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin" />
              <span>문맥 신경망 분석 중...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              <span>AI 문장 후보 생성</span>
            </>
          )}
        </button>
      )}

      {/* Candidate sentence picker */}
      <AnimatePresence>
        {candidates && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-2.5"
          >
            <p className="text-xs font-bold text-neutral-550 mb-1">추천 문장 후보 (신뢰도 순)</p>

            {candidates.map((c, idx) => {
              const isSelected = selectedIndex === idx;
              return (
                <button
                  key={idx}
                  type="button"
                  onClick={() => setSelectedIndex(idx)}
                  className={`w-full text-left p-3.5 rounded-xl border transition-all flex items-center justify-between gap-3 cursor-pointer ${
                    isSelected
                      ? 'bg-brand-green/10 border-brand-green ring-1 ring-brand-green shadow-xs'
                      : 'bg-white border-neutral-200 hover:border-neutral-300 hover:bg-neutral-50'
                  }`}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <div
                      className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 text-[10px] font-bold ${
                        isSelected ? 'bg-brand-green text-white' : 'bg-neutral-100 text-neutral-500'
                      }`}
                    >
                      {isSelected ? <Check className="w-3.5 h-3.5" /> : idx + 1}
                    </div>
                    <span className="font-bold text-sm text-on-surface truncate">{c.text}</span>
                  </div>
                  <span
                    className={`text-[10px] font-bold px-2 py-0.5 rounded-md shrink-0 ${
                      isSelected ? 'bg-brand-green text-white' : 'bg-neutral-100 text-neutral-500'
                    }`}
                  >
                    {c.confidence}%
                  </span>
                </button>
              );
            })}

            <div className="flex gap-2 pt-1">
              <button
                type="button"
                onClick={executeUnkSolve}
                className="flex items-center gap-1.5 px-3.5 py-2.5 bg-neutral-100 hover:bg-neutral-200 text-neutral-600 rounded-xl text-xs font-bold transition-colors cursor-pointer"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                다시 생성
              </button>
              <button
                type="button"
                onClick={handleConfirm}
                disabled={selectedIndex === null}
                className="
                  flex-1 flex items-center justify-center gap-1.5 py-2.5
                  bg-brand-green hover:bg-emerald-600
                  disabled:bg-neutral-100 disabled:text-neutral-400
                  text-white rounded-xl font-bold text-sm transition-all cursor-pointer shadow-sm
                "
              >
                <ChevronRight className="w-4 h-4" />
                이 문장으로 확정
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {confirmed && candidates && selectedIndex !== null && (
        <div className="mt-4 p-4 bg-emerald-50/50 rounded-xl border border-emerald-200/50 animate-in fade-in slide-in-from-top-2 duration-300">
          <p className="text-[10px] text-neutral-450 font-bold uppercase tracking-wider mb-1">
            확정된 번역문 (신뢰도: {candidates[selectedIndex].confidence}%)
          </p>
          <p className="font-bold text-sm text-emerald-950 font-hyper tracking-tight leading-relaxed">
            {candidates[selectedIndex].text}
          </p>
        </div>
      )}
    </div>
  );
};