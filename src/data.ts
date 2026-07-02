import { TranslationLog, GesturePreset } from './types';

export const INITIAL_TRANSLATION_LOGS: TranslationLog[] = [
  {
    log_id: 1,
    medical_id: 1001,
    patient_id: 'none',
    input_time: '2026-05-29T14:32:00.000Z',
    gloss_result: '머리 아프다 심하다',
    translated_text: '머리가 아파서 왔습니다.',
    confidence: 96,
    category: '두통'
  },
  {
    log_id: 2,
    medical_id: 1001,
    patient_id: 'none',
    input_time: '2026-05-29T13:15:00.000Z',
    gloss_result: '기침 일주일 계속',
    translated_text: '일주일 전부터 기침이 계속 나요.',
    confidence: 92,
    category: '호흡기'
  },
  {
    log_id: 3,
    medical_id: 1001,
    patient_id: 'none',
    input_time: '2026-05-28T16:48:00.000Z',
    gloss_result: '열 나다 몸 쑤시다',
    translated_text: '어제부터 열이 나고 몸이 쑤셔요.',
    confidence: 94,
    category: '전신/감기'
  },
  {
    log_id: 4,
    medical_id: 1001,
    patient_id: 'none',
    input_time: '2026-05-28T10:20:00.000Z',
    gloss_result: '배 너무 아프다 잠 못 자다',
    translated_text: '배가 너무 아파서 잠을 못 잤어요.',
    confidence: 89,
    category: '소화기'
  },
  {
    log_id: 5,
    medical_id: 1001,
    patient_id: 'none',
    input_time: '2026-05-27T15:05:00.000Z',
    gloss_result: '오른쪽 팔 아프다',
    translated_text: '오른쪽 팔에 통증이 있습니다.',
    confidence: 91,
    category: '근골격계'
  },
  {
    log_id: 6,
    medical_id: 1001,
    patient_id: 'none',
    input_time: '2026-05-27T11:30:00.000Z',
    gloss_result: '약 먹다 두드러기 나다',
    translated_text: '약을 먹고 나서 두드러기가 났어요.',
    confidence: 85,
    category: '알레르기'
  },
  {
    log_id: 7,
    medical_id: 1001,
    patient_id: 'none',
    input_time: '2026-05-26T09:55:00.000Z',
    gloss_result: '목 붓다 음식 삼키다 어렵다',
    translated_text: '목이 부어서 음식을 삼키기 어려워요.',
    confidence: 95,
    category: '이비인후과'
  },
  {
    log_id: 8,
    medical_id: 1001,
    patient_id: 'none',
    input_time: '2026-05-26T08:40:00.000Z',
    gloss_result: '허리 삐끗',
    translated_text: '허리를 삐끗한 것 같아요.',
    confidence: 88,
    category: '근골격계'
  }
];

export const GESTURE_PRESETS: GesturePreset[] = [
  {
    id: 'g-headache',
    gestureName: '이마에 손 얹기 (forehead touch)',
    translationText: '머리가 아파서 왔습니다. 욱신거리는 통증이 있어요.',
    category: '두통',
    description: '편두통이나 고열의 머리 통증을 나타내는 수어 제스처'
  },
  {
    id: 'g-cough',
    gestureName: '가슴 근처 가볍게 두드리기 (chest tap)',
    translationText: '일주일 전부터 기침이 계속 나고, 가래가 낍니다.',
    category: '호흡기',
    description: '인후통과 기침 증상을 설명하는 호흡기 계통 수어'
  },
  {
    id: 'g-heart',
    gestureName: '심장 부위 움켜쥐기 (tight chest)',
    translationText: '가슴이 답답하고 숨쉬기가 조금 힘듭니다.',
    category: '호흡기',
    description: '호흡 곤란이나 흉통을 직관적으로 표현하는 구급 수어'
  },
  {
    id: 'g-fever',
    gestureName: '목덜미 감싸쥐기 (neck touch)',
    translationText: '어제 밤 이후로 온몸에 오한이 들고 쑤시면서 고열이 나요.',
    category: '전신/감기',
    description: '전신 몸살과 감기 증상을 가리키는 일반적 수소화'
  },
  {
    id: 'g-stomach',
    gestureName: '복부 감싸 안기 (clutching stomach)',
    translationText: '윗배 쪽이 너무 콕콕 찌르듯 아파서 밤새 잠을 못 잤어요.',
    category: '소화기',
    description: '위장 장애나 복통의 급격한 발현을 나타내는 소화기 수어'
  },
  {
    id: 'g-arm',
    gestureName: '한쪽 팔 부위 어루만지기 (arm rub)',
    translationText: '오른쪽 팔꿈치 관절 부근에 강한 찌릿한 통증이 있습니다.',
    category: '근골격계',
    description: '팔부위 타박상 또는 관절염 등을 표시할 때 쓰는 짚기 수어'
  },
  {
    id: 'g-cough2',
    gestureName: '가래 뱉는 시늉 (throat clearing gesture)',
    translationText: '목이 벌겋게 부은 느낌이 들고 침을 삼키기 조차 너무 아픕니다.',
    category: '이비인후과',
    description: '편도선염 등 목 통증 시 사용하는 지시 수어'
  },
  {
    id: 'g-back',
    gestureName: '손등으로 허리 받치기 (hand on lower back)',
    translationText: '무거운 물건을 들다가 허리를 심하게 삐끗해서 굽히기 어렵습니다.',
    category: '근골격계',
    description: '요통 및 디스크 증상을 호소할 때 긴급 전달하는 수어'
  },
  {
    id: 'g-unk',
    gestureName: '정의되지 않은 모호한 손동작 (unk)',
    translationText: 'unk',
    category: '알 수 없음',
    description: '번역 시스템이 의미를 해석하지 못한 비정형적 또는 모호한 손동작'
  }
];