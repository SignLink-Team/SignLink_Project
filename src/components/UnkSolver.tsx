/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Sparkles, HelpCircle, ArrowRight, ShieldAlert, Cpu, Check, Play, RefreshCw, Layers } from 'lucide-react';
import { GLOSS_MAPPING } from '../glossMapping';

interface UnkSolverProps {
  onSolveComplete: (resolvedText: string) => void;
}

interface SelectedPreset {
  id: string;
  glosses: string[];
  candidates: string[];
  correctText: string;
  description: string;
}

const SOLVER_PRESETS: SelectedPreset[] = [
  {
    id: 'breast-cancer',
    glosses: ['유방암', 'UNK', 'UNK', '있다'],
    candidates: ['비염', '진단을받은', '적', '코2', '콧물2', '옷1', '수술1', '검사1'],
    correctText: '유방암 진단을 받은 적 있다',
    description: '수어 문맥 상 유방암(암) 유무 및 병력 청취'
  },
  {
    id: 'fever-cold',
    glosses: ['감기1', 'UNK', '시작1'],
    candidates: ['이마1', '목걸이2', '어제부터', '기억1', '불소', '바지1'],
    correctText: '감기 어제부터 시작',
    description: '감기나 고열 증상의 최초 발현 시점 유추'
  },
  {
    id: 'chest-pain',
    glosses: ['가슴1', 'UNK', '아프다2'],
    candidates: ['콕콕찌르듯', '신발1', '노랑1', '샤워3', '전문1', '바셀린'],
    correctText: '가슴 콕콕찌르듯 아프다',
    description: '호흡기/심장 통증의 디테일한 양상 유추'
  }
];

export const UnkSolver: React.FC<UnkSolverProps> = ({ onSolveComplete }) => {
  const [activePreset, setActivePreset] = useState<SelectedPreset>(SOLVER_PRESETS[0]);
  const [customGlowList, setCustomGlowList] = useState<string>(SOLVER_PRESETS[0].glosses.join(', '));
  const [customCandidates, setCustomCandidates] = useState<string>(SOLVER_PRESETS[0].candidates.join(', '));
  
  // Interactive choosing states
  const [candidatePool, setCandidatePool] = useState<string[]>(SOLVER_PRESETS[0].candidates);
  const [selectedCandidates, setSelectedCandidates] = useState<string[]>(SOLVER_PRESETS[0].candidates);
  const [manualUnkMappings, setManualUnkMappings] = useState<Record<number, string>>({});
  const [selectedUnkIndex, setSelectedUnkIndex] = useState<number | null>(null);
  const [customWordInput, setCustomWordInput] = useState<string>('');

  const [isSolving, setIsSolving] = useState<boolean>(false);
  const [solveStep, setSolveStep] = useState<number>(0);
  const [resolvedSentence, setResolvedSentence] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number>(0);

  // Parse the current simulation inputs
  const currentGlosses = customGlowList.split(',').map(s => s.trim()).filter(Boolean);
  const currentCandidates = customCandidates.split(',').map(s => s.trim()).filter(Boolean);

  const handleSelectPreset = (preset: SelectedPreset) => {
    setActivePreset(preset);
    setCustomGlowList(preset.glosses.join(', '));
    setCustomCandidates(preset.candidates.join(', '));
    setCandidatePool(preset.candidates);
    setSelectedCandidates(preset.candidates);
    setManualUnkMappings({});
    setSelectedUnkIndex(null);
    setResolvedSentence(null);
    setSolveStep(0);
  };

  const executeUnkSolve = () => {
    if (isSolving) return;
    setIsSolving(true);
    setSolveStep(1);
    setResolvedSentence(null);

    // Dynamic timeout simulations to show the advanced pipeline steps
    setTimeout(() => {
      setSolveStep(2); // Step 2: Extracting prefix/suffix context
      
      setTimeout(() => {
        setSolveStep(3); // Step 3: Mapping candidate semantics
        
        setTimeout(() => {
          // Final Resolution
          let resultText = '';
          const isPresetMatch = SOLVER_PRESETS.find(p => p.id === activePreset.id);
          
          let unkCounter = 0;
          let poolCopy = [...selectedCandidates];
          
          const words = currentGlosses.map((g, idx) => {
            if (g.toUpperCase() === 'UNK') {
              const mapped = manualUnkMappings[unkCounter];
              unkCounter++;
              if (mapped) return mapped;
              return poolCopy.shift() || '진단';
            }
            return g;
          });
          
          resultText = words.join(' ');

          // If they haven't manually mapped and they match preset, keep preset exact response
          if (Object.keys(manualUnkMappings).length === 0 && isPresetMatch && customGlowList === isPresetMatch.glosses.join(', ')) {
            resultText = isPresetMatch.correctText;
          }

          setResolvedSentence(resultText);
          setConfidence(Math.round(92 + Math.random() * 6));
          setSolveStep(4);
          setIsSolving(false);
          
          // Bubble translated text up to App level
          onSolveComplete(resultText);
        }, 1200);
      }, 1000);
    }, 1000);
  };

  const unkCount = currentGlosses.filter(g => g.toUpperCase() === 'UNK').length;

  return (
   <div className="w-full bg-white border border-neutral-150 rounded-2xl p-5 md:p-6 shadow-sm">

  {/* Header & Presets */}
  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-5 border-b border-neutral-100 pb-4">
    <div className="flex items-center gap-2">
      <Cpu className="w-5 h-5 text-brand-green animate-pulse"/>
      <div>
        <h3 className="font-bold text-sm text-on-surface">
          AI 문맥 복원 엔진
        </h3>
        <p className="text-[10px] text-neutral-400 font-medium">인식 불가(UNK) 요소를 문맥 맞춤 교정합니다.</p>
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


  {/* Gloss sequence */}
  <div className="bg-neutral-50 border border-neutral-150 rounded-xl p-4 mb-4">
    <div className="flex items-center justify-between mb-2.5">
      <p className="text-xs font-bold text-neutral-550 uppercase tracking-wider">
        실시간 수어 인식 글로스
      </p>
      {selectedUnkIndex !== null && (
        <span className="text-[10px] bg-brand-green text-white font-bold px-2 py-0.5 rounded-md animate-pulse">
          UNK #{selectedUnkIndex + 1} 대입 대기 중
        </span>
      )}
    </div>

    <div className="flex flex-wrap gap-2">
      {(() => {
        let unkCounter = 0;
        return currentGlosses.map((g, idx) => {
          if (g.toUpperCase() === "UNK") {
            const currentUnkNum = unkCounter;
            unkCounter++;
            const mappedWord = manualUnkMappings[currentUnkNum];
            const isSelected = selectedUnkIndex === currentUnkNum;
            
            return (
              <button
                key={idx}
                type="button"
                onClick={() => setSelectedUnkIndex(isSelected ? null : currentUnkNum)}
                className={`px-3 py-1.5 rounded-lg border text-xs font-bold transition-all flex items-center gap-1.5 cursor-pointer ${
                  isSelected
                    ? 'bg-brand-green text-white border-brand-green shadow-xs'
                    : mappedWord
                    ? 'bg-emerald-50 text-brand-green border-brand-green/35 hover:bg-emerald-100/50'
                    : 'bg-red-50 text-red-600 border-red-250 hover:bg-red-100/30'
                }`}
              >
                <span>{mappedWord ? `UNK #${currentUnkNum + 1}: ${mappedWord}` : `UNK #${currentUnkNum + 1} 선택`}</span>
                <span className="text-[10px] opacity-75">
                  {isSelected ? '●' : '✏️'}
                </span>
              </button>
            );
          }

          return (
            <span
              key={idx}
              className="px-3 py-1.5 rounded-lg bg-white border border-neutral-200 text-neutral-700 text-xs font-bold shadow-3xs"
            >
              {g}
            </span>
          );
        });
      })()}
    </div>

    <p className="text-[11px] text-neutral-400 mt-3 font-medium">
      💡 매핑할 UNK 슬롯을 지정하고(혹은 순서대로) 아래 단어를 선택해 온전한 문장을 이끌어내세요.
    </p>
  </div>


  {/* Candidate words */}
  <div className="space-y-2.5">
    <div className="flex items-center justify-between">
      <p className="text-xs font-bold text-neutral-550">
        UNK 추천 후보 단어군
      </p>
      {Object.keys(manualUnkMappings).length > 0 && (
        <button
          type="button"
          onClick={() => {
            setManualUnkMappings({});
            setSelectedUnkIndex(null);
          }}
          className="text-[10px] text-red-500 hover:text-red-600 font-bold cursor-pointer transition-colors"
        >
          초기화
        </button>
      )}
    </div>

    <div className="flex flex-wrap gap-1.5 max-h-56 overflow-y-auto p-1 bg-neutral-50/50 rounded-xl border border-neutral-100">
      {candidatePool.map(word => {
        const assignedSlotIndexStr = Object.keys(manualUnkMappings).find(key => manualUnkMappings[Number(key)] === word);
        const isAssigned = assignedSlotIndexStr !== undefined;
        
        return (
          <button
            key={word}
            type="button"
            onClick={() => {
              let targetSlotIndex = selectedUnkIndex;
              if (targetSlotIndex === null) {
                // Find the first unassigned slot
                for (let i = 0; i < unkCount; i++) {
                  if (manualUnkMappings[i] === undefined) {
                    targetSlotIndex = i;
                    break;
                  }
                }
                if (targetSlotIndex === null) {
                  targetSlotIndex = 0;
                }
              }

              setManualUnkMappings(prev => ({
                ...prev,
                [targetSlotIndex!]: word
              }));
              setSelectedUnkIndex(null);
            }}
            className={`px-3 py-1.5 rounded-lg border text-xs font-bold transition-all cursor-pointer ${
              isAssigned
                ? 'bg-brand-green/10 text-brand-green border-brand-green/45 font-extrabold shadow-3xs'
                : 'bg-white hover:bg-neutral-50 hover:border-neutral-350 text-neutral-600 border-neutral-200 shadow-3xs'
            }`}
          >
            <span>{word}</span>
            {isAssigned && (
              <span className="text-[9px] ml-1.5 text-brand-green/80 font-medium">
                (UNK #{Number(assignedSlotIndexStr) + 1})
              </span>
            )}
          </button>
        );
      })}
    </div>

    {/* Custom candidate addition */}
    <div className="flex gap-2">
      <input
        type="text"
        placeholder="수동 후보 단어 추가..."
        value={customWordInput}
        onChange={(e) => setCustomWordInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            const val = customWordInput.trim();
            if (val && !candidatePool.includes(val)) {
              setCandidatePool(prev => [...prev, val]);
              setCustomWordInput('');
            }
          }
        }}
        className="flex-1 px-3 py-1.5 bg-neutral-50 hover:bg-neutral-100/50 focus:bg-white border border-neutral-200 focus:border-brand-green focus:ring-1 focus:ring-brand-green/30 rounded-lg text-xs font-semibold outline-none transition-all shadow-3xs"
      />
      <button
        type="button"
        onClick={() => {
          const val = customWordInput.trim();
          if (val && !candidatePool.includes(val)) {
            setCandidatePool(prev => [...prev, val]);
            setCustomWordInput('');
          }
        }}
        className="px-3 py-1.5 bg-brand-green hover:bg-emerald-600 text-white font-bold rounded-lg text-xs transition-colors cursor-pointer shadow-3xs"
      >
        추가
      </button>
    </div>
  </div>


  <button
    type="button"
    onClick={executeUnkSolve}
    disabled={isSolving}
    className="
    mt-5 w-full py-3
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
        <span>실시간 AI 소견 복원 완료하기</span>
      </>
    )}
  </button>


  {resolvedSentence && (
    <div className="
      mt-4 p-4
      bg-emerald-50/50
      rounded-xl
      border border-emerald-200/50
      animate-in fade-in slide-in-from-top-2 duration-300
    ">
      <p className="text-[10px] text-neutral-450 font-bold uppercase tracking-wider mb-1">
        복원 완료된 번역문 (신뢰 유사도: {confidence}%)
      </p>

      <p className="font-bold text-sm text-emerald-950 font-hyper tracking-tight leading-relaxed">
        {resolvedSentence}
      </p>
    </div>
  )}

</div>
  );
};
