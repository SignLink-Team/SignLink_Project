import { GesturePreset } from './types'

export const GESTURE_PRESETS: GesturePreset[] = [
  {
    id: 'g-headache',
    gestureName: '이마에 손 얹기',
    translationText: '머리가 아파서 왔습니다. 욱신거리는 통증이 있어요.',
    category: '두통',
    description: '두통과 어지러움을 표현하는 수어 제스처',
  },
  {
    id: 'g-cough',
    gestureName: '가슴 근처 가볍게 두드리기',
    translationText: '일주일 전부터 기침이 계속 나요.',
    category: '호흡기',
    description: '기침과 호흡기 증상을 설명하는 수어',
  },
  {
    id: 'g-heart',
    gestureName: '가슴 부위 움켜쥐기',
    translationText: '가슴이 답답하고 숨쉬기가 조금 힘듭니다.',
    category: '호흡기',
    description: '흉통이나 호흡 불편감을 전달하는 수어',
  },
  {
    id: 'g-fever',
    gestureName: '목덜미 감싸쥐기',
    translationText: '어제부터 열이 나고 몸이 쑤셔요.',
    category: '전신/감기',
    description: '발열과 몸살을 나타내는 수어',
  },
  {
    id: 'g-stomach',
    gestureName: '복부 감싸 안기',
    translationText: '배가 너무 아파서 잠을 못 잤어요.',
    category: '소화기',
    description: '복통과 소화기 불편감을 표현하는 수어',
  },
  {
    id: 'g-arm',
    gestureName: '한쪽 팔 부위 어루만지기',
    translationText: '오른쪽 팔에 통증이 있습니다.',
    category: '근골격계',
    description: '팔과 관절 통증을 전달하는 수어',
  },
  {
    id: 'g-allergy',
    gestureName: '피부를 긁는 동작',
    translationText: '약을 먹고 나서 두드러기가 났어요.',
    category: '알레르기',
    description: '피부 가려움과 알레르기를 표현하는 수어',
  },
  {
    id: 'g-throat',
    gestureName: '목을 감싸는 동작',
    translationText: '목이 부어서 음식을 삼키기 어려워요.',
    category: '이비인후과',
    description: '목 통증과 삼킴 곤란을 표현하는 수어',
  },
]
