
export type ViewState = 'main' | 'history' | 'login';

export interface TranslationLog {
  log_id: number;
  medical_id: number;       // 로그인된 의료진 user_id
  patient_id: string;       // 환자 식별번호
  input_time: string;       // 번역 요청 시각 (ISO datetime string)
  gloss_result: string;     // 인식된 수어 글로스 시퀀스
  translated_text: string;  // 최종 번역 문장
  confidence: number;       // 번역 신뢰도 (0~100 또는 0~1, 백엔드 기준에 맞춰 사용)
  category?: string;        // 진료 분류 카테고리 (필터링 유지용)
  isCustom?: boolean;       // 직접 수동 입력 여부
}

export interface UserState {
  isLoggedIn: boolean;
  email: string | null;
  name: string | null;
  userId?: number;
}

export interface GesturePreset {
  id: string;
  gestureName: string;
  translationText: string;
  category: string;
  description: string;
}
