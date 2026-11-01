// CameraView의 모션 게이트/자동 캘리브레이션에 쓰는 상수와 순수 함수.

export const CAMERA_EXTRACTION_FPS = 30
export const CAMERA_FRAME_INTERVAL_MS = 1000 / CAMERA_EXTRACTION_FPS

// 최소 샘플 수와 실제 경과 시간을 모두 만족한 뒤 보정을 완료한다.
// 프레임 속도가 빨라도 안내를 읽고 정지할 시간을 확보한다.
export const CALIBRATION_FRAMES = 20
export const CALIBRATION_MIN_DURATION_MS = 2000
// 보정 baseline으로 쓰는 하위 백분위(20%)
export const CALIBRATION_BASELINE_PERCENTILE = 0.2

// keypointExtractor의 포즈 블록: 33개 랜드마크 × 3개 좌표.
// 이 블록이 전부 0이면 사람이 검출되지 않은 프레임이다.
export const POSE_KEYPOINT_VALUES = 99
// 손 데이터는 손당 78개
// 21개 랜드마크 × xyz(63개) + 관절각(15개)
export const HAND_KEYPOINT_VALUES = 78
export const HAND_COORDINATE_VALUES = 21 * 3
export const KEYPOINT_VECTOR_LENGTH = 261

export const LEFT_HAND_START = POSE_KEYPOINT_VALUES
export const RIGHT_HAND_START = LEFT_HAND_START + HAND_KEYPOINT_VALUES

// 추정 초기값. 실제 카메라/추출기에서 측정한 정지 잡음에 맞춰 조정한다.
// baseline(기존 20백분위값)이 이 값을 초과하면 보정을 완료하지 않는다.
export const MAX_CALIBRATION_BASELINE = 0.02
// 상반신/손이 이 프레임 수 이상 계속 안 보이면 "무엇이 안 보이는지"를 안내한다(검출 지연으로 문구가 깜빡이지 않게).
export const MISSING_NOTICE_FRAMES = 10
export const START_THRESHOLD_MULTIPLIER = 6
export const END_THRESHOLD_MULTIPLIER = 2.5
export const MIN_START_THRESHOLD = 0.0015
export const MIN_END_THRESHOLD = 0.0006

export const MOTION_START_FRAMES = 3
export const WORD_END_FRAMES = 8
export const SENTENCE_END_FRAMES = 30
export const POST_END_COOLDOWN_FRAMES = 8

// 수어 종료 방식
//  true : 모션이 완전히 멈춰도 문장을 끝내지 않는다. 사용자가 "수어 종료" 버튼을 눌렀을 때만
//         스트림을 종료하고 녹화 전체를 번역한다.
//         (정지 중 프레임 전송 중단과 단어 경계 전송은 기존 그대로 유지한다)
//  false: 기존처럼 SENTENCE_END_FRAMES만큼 정지하면 자동 종료한다.
export const MANUAL_STOP_MODE: boolean = true
// 수동 모드에서 버튼을 잊고 방치했을 때의 안전장치. 이 프레임 수만큼 계속 정지하면 자동 종료한다
// (30fps 기준 30초). 0이면 안전장치를 끄고 오직 버튼으로만 종료한다.
export const IDLE_SAFETY_END_FRAMES: number = 900
// 한 세션에서 보낸 프레임이 이 수에 도달하면 자동 종료한다. 서버의 MAX_RECORD_FRAMES(5400)와 같은 값.
export const MAX_MANUAL_RECORD_FRAMES = 5400
// 버튼을 누르러 손을 뻗는 동작이 새 수어 세션으로 오인되지 않게 하는 쿨다운(30fps 기준 1.5초)
export const MANUAL_STOP_COOLDOWN_FRAMES = 45

/** 직전 프레임 대비 좌표 변화량의 평균. 비교할 수 없으면 0. */
export function computeMotionScore(prev: number[] | null, curr: number[]): number {
  if (!prev || prev.length !== curr.length || curr.length === 0) {
    return 0
  }

  let sum = 0

  for (let i = 0; i < curr.length; i += 1) {
    sum += Math.abs(curr[i] - prev[i])
  }

  return sum / curr.length
}

/** 추출 결과가 길이와 값(NaN/Infinity 없음) 모두 유효한 keypoint 벡터인지 확인 */
export function isValidKeypointVector(keypoints: number[] | null): keypoints is number[] {
  return (
    !!keypoints &&
    keypoints.length >= KEYPOINT_VECTOR_LENGTH &&
    keypoints.every((value) => Number.isFinite(value))
  )
}

/** 포즈(상반신) 블록에 0이 아닌 값이 하나라도 있으면 사람이 검출된 것으로 본다. */
export function hasDetectedPose(keypoints: number[]): boolean {
  return (
    keypoints.length >= POSE_KEYPOINT_VALUES &&
    keypoints.slice(0, POSE_KEYPOINT_VALUES).some((value) => value !== 0)
  )
}

/**
 * 손 랜드마크 검출 여부 확인
 *
 * keypointExtractor.ts에서 손이 검출되지 않으면
 * 해당 손의 78개 데이터가 모두 0으로 반환된다.
 *
 * 좌표 63개를 검사하여 손 검출 여부를 판단한다.
 */
function hasDetectedHand(keypoints: number[], startIndex: number): boolean {
  const coordinates = keypoints.slice(startIndex, startIndex + HAND_COORDINATE_VALUES)

  if (coordinates.length !== HAND_COORDINATE_VALUES) {
    return false
  }

  return coordinates.some((value) => Math.abs(value) > 1e-6)
}

/** 왼손 또는 오른손 중 하나라도 검출되었는지 확인 */
export function hasDetectedAnyHand(keypoints: number[]): boolean {
  return hasDetectedHand(keypoints, LEFT_HAND_START) || hasDetectedHand(keypoints, RIGHT_HAND_START)
}

/** 정렬된 보정 샘플에서 하위 백분위 값을 baseline으로 고른다. */
export function computeCalibrationBaseline(samples: number[]): number {
  const sorted = [...samples].sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length * CALIBRATION_BASELINE_PERCENTILE)]
}

/** 수어 세션 중 자동으로 문장을 끝내야 하는지 판단 */
export function shouldAutoEndSentence(stillFrames: number, sentFrames: number): boolean {
  if (MANUAL_STOP_MODE) {
    return (
      (IDLE_SAFETY_END_FRAMES > 0 && stillFrames >= IDLE_SAFETY_END_FRAMES) ||
      sentFrames >= MAX_MANUAL_RECORD_FRAMES
    )
  }
  return stillFrames >= SENTENCE_END_FRAMES
}

/** MediaStream의 모든 트랙을 정지한다. */
export function stopMediaStream(stream: MediaStream | null | undefined) {
  stream?.getTracks().forEach((track) => track.stop())
}
