interface TTSToggleProps {
  enabled: boolean;
  onToggle: () => void;
  isSpeaking: boolean;
}

export function TTSToggle({ enabled, onToggle, isSpeaking }: TTSToggleProps) {
  return (
    <button
      onClick={onToggle}
      className={`flex items-center gap-2 px-3 py-1.5 rounded-full transition-colors
        ${enabled ? 'bg-brand-green text-white font-bold' : 'bg-gray-200 text-gray-500 font-bold'}`}
    >
      {enabled ? '🔊' : '🔇'}
      <span className="text-sm">{!enabled ? '음성으로 듣기' : isSpeaking ? '재생 중' : '자동 재생 켜짐'}</span>
    </button>
  );
}