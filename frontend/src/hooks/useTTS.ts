import { useState, useCallback, useRef, useEffect } from 'react';

// 브라우저/OS별 한국어 음성 우선순위 (자연스러운 순)
const PREFERRED_VOICE_NAMES = [
  'Google 한국의',
  'Microsoft SunHi Online (Natural) - Korean (Korea)',
  'Microsoft Heami - Korean (Korea)',
  'Yuna',
];

function pickBestKoreanVoice(voices: SpeechSynthesisVoice[]): SpeechSynthesisVoice | null {
  const koreanVoices = voices.filter(v => v.lang.startsWith('ko'));
  if (koreanVoices.length === 0) return null;

  for (const preferredName of PREFERRED_VOICE_NAMES) {
    const match = koreanVoices.find(v => v.name.includes(preferredName));
    if (match) return match;
  }
  // 우선순위에 없으면 첫 번째 한국어 음성이라도 사용
  return koreanVoices[0];
}

// 숫자/단위 등을 자연스러운 한국어 발음으로 정규화
function normalizeForSpeech(text: string): string {
  return text
    .replace(/(\d+)\.(\d+)/g, '$1점$2') // 37.5 → 37점5
    .replace(/(\d+)%/g, '$1퍼센트')
    .replace(/(\d+)mg/gi, '$1밀리그램')
    .replace(/(\d+)kg/gi, '$1킬로그램')
    .replace(/(\d+)cm/gi, '$1센티미터')
    .trim();
}

// 문장 단위로 쪼개서 순차 재생 (긴 문장 잘림/뭉개짐 방지)
function splitSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+|(?<=[다요])\s(?=[가-힣])/)
    .map(s => s.trim())
    .filter(Boolean);
}

export function useTTS() {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voice, setVoice] = useState<SpeechSynthesisVoice | null>(null);
  const queueRef = useRef<string[]>([]);

  useEffect(() => {
    const loadVoice = () => {
      const best = pickBestKoreanVoice(window.speechSynthesis.getVoices());
      if (best) setVoice(best);
    };
    loadVoice();
    window.speechSynthesis.onvoiceschanged = loadVoice;
    return () => {
      window.speechSynthesis.onvoiceschanged = null;
    };
  }, []);

  const speakNext = useCallback(() => {
    const nextText = queueRef.current.shift();
    if (!nextText) {
      setIsSpeaking(false);
      return;
    }
    const utter = new SpeechSynthesisUtterance(nextText);
    utter.lang = 'ko-KR';
    if (voice) utter.voice = voice;
    utter.rate = 0.95;  // 병원 안내용이라 살짝 느리게
    utter.pitch = 1.0;
    utter.onend = speakNext; // 큐에 남은 다음 문장 이어서 재생
    window.speechSynthesis.speak(utter);
  }, [voice]);

  const speak = useCallback((text: string) => {
    window.speechSynthesis.cancel();
    const normalized = normalizeForSpeech(text);
    queueRef.current = splitSentences(normalized);
    setIsSpeaking(true);
    speakNext();
  }, [speakNext]);

  const stop = useCallback(() => {
    queueRef.current = [];
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  }, []);

  return { speak, stop, isSpeaking, voice };
}