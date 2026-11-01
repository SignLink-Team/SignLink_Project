import { GesturePreset, TranslationLog } from './types'

export const INITIAL_TRANSLATION_LOGS: TranslationLog[] = []

export const MEDICAL_CATEGORIES = [
  '전체',
  '두통',
  '호흡기',
  '전신/감기',
  '소화기',
  '근골격계',
  '알레르기',
  '이비인후과',
  '기타',
]

export const GESTURE_PRESETS: GesturePreset[] = [
  {
    id: 'g-headache',
    gestureName: '이마에 손 얹기',
    translationText: '머리가 아파서 왔습니다. 욱신거리는 통증이 있어요.',
    category: '두통',
    description: '두통과 어지러움 증상을 표현하는 테스트 수어',
  },
  {
    id: 'g-cough',
    gestureName: '가슴 근처 두드리기',
    translationText: '일주일 전부터 기침이 계속 나고 가래가 낍니다.',
    category: '호흡기',
    description: '기침과 호흡기 증상을 표현하는 테스트 수어',
  },
  {
    id: 'g-fever',
    gestureName: '목덜미 감싸기',
    translationText: '어제부터 열이 나고 몸이 으슬으슬합니다.',
    category: '전신/감기',
    description: '발열과 감기 증상을 표현하는 테스트 수어',
  },
  {
    id: 'g-stomach',
    gestureName: '복부 감싸기',
    translationText: '배가 너무 아파서 잠을 잘 못 잤어요.',
    category: '소화기',
    description: '복통과 소화기 증상을 표현하는 테스트 수어',
  },
  {
    id: 'g-arm',
    gestureName: '팔 부위 문지르기',
    translationText: '오른쪽 팔에 통증이 있습니다.',
    category: '근골격계',
    description: '근골격계 통증을 표현하는 테스트 수어',
  },
  {
    id: 'g-unk',
    gestureName: '모호한 동작',
    translationText: '유방암 UNK UNK 있다',
    category: '기타',
    description: 'UNK 후보 문장 선택 화면 확인용 테스트 수어',
  },
]
