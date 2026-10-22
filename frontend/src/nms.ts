// 비수지(NMS) 공용 타입/상수.
// 서버(ai_server/main.py)의 NMS_KEYS / NMS_KEY_THRESHOLD와 값·순서를 항상 같게 유지할 것.

export const NMS_KEYS = ['Mo1', 'Mmo', 'Mctr', 'Ci', 'Ebu', 'EBf', 'Hno', 'Hs'] as const
export type NmsKey = (typeof NMS_KEYS)[number]

export const NMS_KEY_THRESHOLD: Record<NmsKey, number> = {
  Mo1: 0.42,
  Mmo: 0.42,
  Mctr: 0.42,
  Ci: 0.4,
  Ebu: 0.45,
  EBf: 0.52,
  Hno: 0.5,
  Hs: 0.4,
}

// 기존 OpenCV 스크립트(BGR)의 색을 RGB로 변환한 값
export const NMS_KEY_COLORS: Record<NmsKey, string> = {
  Mo1: 'rgb(255, 100, 50)',
  Mmo: 'rgb(255, 180, 50)',
  Mctr: 'rgb(255, 220, 100)',
  Ci: 'rgb(50, 200, 255)',
  Ebu: 'rgb(50, 255, 150)',
  EBf: 'rgb(100, 180, 255)',
  Hno: 'rgb(200, 80, 255)',
  Hs: 'rgb(255, 80, 150)',
}

export const NMS_FACE_POINTS = 468

/** 한 프레임의 얼굴 랜드마크. landmarks는 468점 x (x, y, z) = 1404개 값. */
export interface FaceFrame {
  landmarks: number[]
  width: number
  height: number
}

/** 서버가 보내는 실시간 비수지 결과 */
export interface NmsFrame {
  frameId: number
  probs: number[]
  smoothed: number[]
  active: boolean[]
}

/** 세션 종료 시 서버가 돌려주는 비수지 구간(초 단위, 세션 시작 기준) */
export interface NmsSegment {
  key: string
  start: number
  end: number
  /** 구간 중 smoothed 확률의 최댓값(신호 강도) */
  peak?: number
}

type Landmark = { x: number; y: number; z?: number }

/**
 * MediaPipe 얼굴 랜드마크(Landmark[] 또는 Landmark[][])를 서버로 보낼 평탄 배열로 변환한다.
 * - 468점만 사용한다(iris 포함 478점이어도 앞 468점만).
 * - 소수 5자리로 반올림해 JSON 크기를 줄인다(프레임당 약 12KB).
 */
export function packFaceLandmarks(
  raw: Landmark[] | Landmark[][] | null | undefined,
): number[] | null {
  if (!raw || raw.length === 0) return null

  const first = raw[0]
  const list = (Array.isArray(first) ? first : raw) as Landmark[]
  if (!list || list.length < NMS_FACE_POINTS) return null

  const round = (v: number) => Math.round(v * 1e5) / 1e5
  const out = new Array<number>(NMS_FACE_POINTS * 3)

  for (let i = 0; i < NMS_FACE_POINTS; i += 1) {
    const p = list[i]
    out[i * 3] = round(p.x)
    out[i * 3 + 1] = round(p.y)
    out[i * 3 + 2] = round(p.z ?? 0)
  }

  return out
}