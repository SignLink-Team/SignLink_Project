import { Fragment } from 'react'
import { ChevronRight, Hand } from 'lucide-react'
import { CameraGuideVariant } from '../../types'

// CameraView의 사용 안내(단계 표시, 사용 방법 카드, 상반신 구도 가이드)와 안내 문구.

// done: 한 문장의 수어를 마친 상태. '다시 맞추기'를 누르기 전까지 새 수어 인식을 시작하지 않는다.
export type MotionCalibrationStatus = 'preparing' | 'calibrating' | 'unstable' | 'ready' | 'done'
/** 보정이 시작되지 못하는 이유(상반신/손 미검출) */
export type MissingPart = 'body' | 'hand' | null

interface GuideText {
  title: string
  description: string
}

// 상반신(머리~가슴) 구도 가이드. 선만 그려 몸을 가리지 않는다. 모양은 도움말 창에서 고를 수 있다.
const GUIDE_WIDTH = 540
const GUIDE_HEIGHT = 491
const GUIDE_SHAPES: Record<
  CameraGuideVariant,
  { head?: { cx: number; cy: number; rx: number; ry: number }; path: string }
> = {
  basic: {
    head: { cx: 270, cy: 125, rx: 92, ry: 118 },
    path: 'M 229 234 C 229 258 219 268 197 276 C 104 296 16 336 6 486 M 311 234 C 311 258 321 268 343 276 C 436 296 524 336 534 486',
  },
  silhouette: {
    path: [
      'M 136 491',
      'C 139 453 145 414 153 379',
      'C 159 353 165 329 176 311',
      'C 187 296 213 292 237 282',
      'C 247 278 250 270 249 259',
      'C 240 254 235 245 233 235',
      'C 228 235 226 230 226 224',
      'C 226 220 228 217 231 217',
      'C 227 204 229 193 237 185',
      'C 245 177 257 174 270 174',
      'C 283 174 295 177 303 185',
      'C 311 193 313 204 309 217',
      'C 312 217 314 220 314 224',
      'C 314 230 312 235 307 235',
      'C 305 245 300 254 291 259',
      'C 290 270 293 278 303 282',
      'C 327 292 353 296 364 311',
      'C 375 329 381 353 387 379',
      'C 395 414 401 453 404 491',
    ].join(' '),
  },
}

export const CAMERA_GUIDE_VARIANTS: Array<{ value: CameraGuideVariant; label: string }> = [
  { value: 'basic', label: '기본형' },
  { value: 'silhouette', label: '윤곽형' },
]

/** 가이드 모양의 선. 카메라 화면과 도움말 창의 미리보기가 함께 쓴다. */
function GuideShape({ variant }: { variant: CameraGuideVariant }) {
  const { head, path } = GUIDE_SHAPES[variant]
  return (
    <>
      {head && <ellipse {...head} vectorEffect="non-scaling-stroke" />}
      <path
        d={path}
        transform={variant === 'silhouette' ? 'translate(270 491) scale(1.04) translate(-270 -491)' : undefined}
        vectorEffect="non-scaling-stroke"
      />
    </>
  )
}

/** 도움말 창에서 가이드 모양을 고를 때 보여주는 작은 미리보기 */
export function GuideShapePreview({ variant }: { variant: CameraGuideVariant }) {
  return (
    <svg
      aria-hidden="true"
      viewBox={`0 0 ${GUIDE_WIDTH} ${GUIDE_HEIGHT}`}
      className="h-11 w-12 shrink-0 stroke-neutral-600"
      fill="none"
      strokeWidth={2}
      strokeLinecap="round"
    >
      <GuideShape variant={variant} />
    </svg>
  )
}

// 구도 가이드는 안내용이다. 실제 얼굴/상체/양손 검출 판정과 모션 게이트는 변경하지 않는다.
const CALIBRATION_GUIDE_TEXT: Record<MotionCalibrationStatus, GuideText> = {
  preparing: {
    title: '상반신을 윤곽에 맞춰 주세요',
    description:
      '머리부터 가슴까지가 윤곽 안에 들어오게 앉고, 두 손은 가슴 앞에 올려 주세요. 맞으면 자동으로 다음 단계로 넘어가요.',
  },
  calibrating: {
    title: '그대로 가만히 계세요',
    description: '약 2초 동안 몸과 손을 움직이지 마세요. 끝나면 바로 수어를 시작할 수 있어요.',
  },
  unstable: {
    title: '움직임이 감지되었어요',
    description: '아래 ‘다시 맞추기’를 누른 뒤, 몸과 손을 움직이지 말고 가만히 계세요.',
  },
  ready: {
    title: '이제 수어를 시작하세요!',
    description: '손을 움직이면 자동으로 인식이 시작돼요. 수어를 마치면 빨간 ‘수어 종료’ 버튼을 눌러 주세요.',
  },
  done: {
    title: '번역 결과를 확인해 주세요',
    description: '다음 수어를 하려면 아래 ‘다시 수어하기’를 눌러 주세요. 자리 맞추기부터 다시 시작해요.',
  },
}

// 보정이 시작되지 못하는 이유를 알면 그 이유를 직접 안내한다(조용히 기다리게 두지 않는다).
const MISSING_PART_TEXT: Record<Exclude<MissingPart, null>, GuideText> = {
  body: {
    title: '상반신이 보이지 않아요',
    description: '카메라 앞에 앉아 머리부터 가슴까지가 윤곽 안에 들어오게 해 주세요.',
  },
  hand: {
    title: '두 손이 보이지 않아요',
    description: '두 손을 가슴 앞으로 올려 카메라에 보여 주세요. 손이 보이면 자동으로 다음 단계로 넘어가요.',
  },
}

const RECORDING_TEXT: GuideText = {
  title: '수어를 인식하고 있어요',
  description: '두 손을 화면 안에 두고 수어를 해 주세요. 다 끝나면 빨간 ‘수어 종료’ 버튼을 눌러 주세요.',
}

const PREDICTING_TEXT: GuideText = {
  title: '번역하고 있어요',
  description: '결과가 나올 때까지 잠시만 기다려 주세요. 카메라는 그대로 켜 두세요.',
}

const STATUS_STEP: Record<MotionCalibrationStatus, number> = {
  preparing: 2,
  calibrating: 3,
  unstable: 3,
  ready: 4,
  // 모든 단계를 마친 상태(단계 표시가 전부 완료로 보인다)
  done: 5,
}

interface GuideState {
  isRecording: boolean
  isPredicting: boolean
  calibrationStatus: MotionCalibrationStatus
  missingPart: MissingPart
}

/** 현재 단계(1~4). 5는 모든 단계를 마치고 번역 결과를 기다리는 중. */
export function getCurrentStep({ isRecording, isPredicting, calibrationStatus }: GuideState): number {
  if (isPredicting) return 5
  if (isRecording) return 4
  return STATUS_STEP[calibrationStatus]
}

/** 영상 아래 안내 패널의 제목/설명 */
export function getGuideText({ isRecording, isPredicting, calibrationStatus, missingPart }: GuideState): GuideText {
  if (isRecording) return RECORDING_TEXT
  if (isPredicting) return PREDICTING_TEXT
  if (calibrationStatus === 'preparing' && missingPart) return MISSING_PART_TEXT[missingPart]
  return CALIBRATION_GUIDE_TEXT[calibrationStatus]
}

// 처음 쓰는 사람을 위한 4단계. 카메라를 켜기 전에는 사용 방법으로, 켠 뒤에는 현재 단계 표시로 쓴다.
const GUIDE_STEPS = [
  { title: '카메라 켜기', hint: '위 화면을 눌러 카메라를 켜요.' },
  {
    title: '자리 맞추기',
    hint: '머리부터 가슴까지가 윤곽 안에 들어오게 앉고, 두 손은 가슴 앞에 올려요.',
  },
  { title: '가만히 있기', hint: '약 2초 동안 움직이지 않고 기다려요.' },
  {
    title: '수어 하기',
    hint: '손을 움직이면 인식이 시작돼요. 다 끝나면 빨간 ‘수어 종료’ 버튼을 눌러요.',
  },
] as const

// current: 1~4는 진행 중인 단계, 5는 모든 단계 완료(번역 중)
// compact: 카메라 영상 위(가이드 머리 위)에 얹을 때 쓰는 작은 크기
export function GuideSteps({ current, compact = false }: { current: number; compact?: boolean }) {
  // 모든 단계를 마치면(번역 중/번역 완료) 진행 방향 화살표는 숨긴다.
  const allDone = current > GUIDE_STEPS.length

  return (
    <ol aria-label="사용 단계" className={`flex items-start ${compact ? 'gap-0.5 sm:gap-1.5' : 'gap-1 sm:gap-2'}`}>
      {GUIDE_STEPS.map((step, index) => {
        const number = index + 1
        const done = number < current
        const active = number === current
        const showArrow = !allDone && number < GUIDE_STEPS.length

        return (
          <Fragment key={step.title}>
            <li
              aria-current={active ? 'step' : undefined}
              className={`flex min-w-0 flex-1 flex-col items-center text-center ${compact ? 'gap-1' : 'gap-1.5'}`}
            >
              {/* 지금 단계는 불이 들어온 것처럼 밝게 빛나고, 지난 단계는 초록색으로 켜진 채 남는다. */}
              <span
                className={`flex items-center justify-center rounded-full font-bold transition-all duration-500 motion-reduce:transition-none ${
                  compact ? 'h-7 w-7 text-sm sm:h-8 sm:w-8 sm:text-base' : 'h-9 w-9 text-base sm:h-10 sm:w-10 sm:text-lg'
                } ${
                  done
                    ? 'bg-emerald-400 text-neutral-950 shadow-[0_0_12px_2px_rgba(52,211,153,0.45)]'
                    : active
                      ? 'bg-white text-neutral-950 ring-4 ring-white/30 shadow-[0_0_18px_6px_rgba(255,255,255,0.5)]'
                      : 'bg-white/10 text-white/60'
                }`}
              >
                <span aria-hidden="true">{done ? '✓' : number}</span>
                <span className="sr-only">{done ? `${number}단계 완료` : `${number}단계`}</span>
              </span>
              <span
                className={`break-keep transition-colors duration-500 motion-reduce:transition-none ${
                  compact ? 'whitespace-nowrap text-[11px] leading-tight sm:text-sm' : 'text-sm leading-snug sm:text-base'
                } ${
                  active ? 'font-bold text-white' : done ? 'font-medium text-emerald-200' : 'text-white/60'
                }`}
              >
                {step.title}
              </span>
            </li>

            {/* 단계 사이 화살표. 앞 단계를 마치면 초록색으로 켜진다. 원의 가운데 높이에 맞춘다. */}
            {showArrow && (
              <li aria-hidden="true" className={`shrink-0 ${compact ? 'mt-1.5 sm:mt-2' : 'mt-2.5 sm:mt-3'}`}>
                <ChevronRight
                  className={`transition-colors duration-500 motion-reduce:transition-none ${compact ? 'h-4 w-4' : 'h-5 w-5'} ${
                    done ? 'text-emerald-300' : 'text-white/25'
                  }`}
                  strokeWidth={3}
                />
              </li>
            )}
          </Fragment>
        )
      })}
    </ol>
  )
}

// 카메라가 꺼져 있을 때 영상 아래에 보여주는 사용 방법 카드
export function HowToCard() {
  return (
    <div className="mt-3 rounded-2xl border border-neutral-800 bg-neutral-950 px-5 py-5 text-white sm:px-7 sm:py-6">
      <h3 className="text-xl font-bold sm:text-2xl">이렇게 사용해요</h3>

      <ol className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
        {GUIDE_STEPS.map((step, index) => (
          <li key={step.title} className="flex gap-3 rounded-xl bg-white/5 p-4">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-white text-lg font-bold text-neutral-950">
              {index + 1}
            </span>
            <div className="min-w-0">
              <p className="text-lg font-bold">{step.title}</p>
              <p className="mt-1 break-keep text-base leading-relaxed text-white/85">{step.hint}</p>
            </div>
          </li>
        ))}
      </ol>
    </div>
  )
}

const HAND_MARKER_CLASS =
  'absolute top-[38%] hidden h-14 w-14 sm:flex animate-pulse items-center justify-center rounded-2xl border-2 border-dashed border-white/90 bg-neutral-950/40 motion-reduce:animate-none sm:h-16 sm:w-16'

/** 상태별 윤곽 선 종류와 영상 위 한 줄 안내. 안내가 없으면(수어 중 등) 윤곽만 표시한다. */
function getPositionGuideAppearance({ isRecording, isPredicting, calibrationStatus, missingPart }: GuideState) {
  if (isPredicting || isRecording || calibrationStatus === 'done' || calibrationStatus === 'ready') {
    return { dashed: false, label: null }
  }
  if (calibrationStatus === 'calibrating') return { dashed: false, label: '좋아요, 그대로 계세요' }
  if (calibrationStatus === 'unstable') return { dashed: true, label: '움직임이 감지되었어요' }
  return {
    dashed: true,
    label: missingPart === 'hand' ? '두 손을 가슴 앞으로 올려 주세요' : '얼굴과 어깨를 선에 맞춰 주세요',
  }
}

/**
 * 구도 안내 전용: 얼굴과 어깨가 선에 맞도록 거리를 맞춘다. 실제 검출 판정은 하지 않는다.
 * 색은 항상 흰색이다. 맞추는 중이거나 다시 맞춰야 하면 점선, 그 외에는 실선으로 표시한다.
 */
export function PositionGuideOverlay({
  state,
  variant = 'basic',
}: {
  state: GuideState
  variant?: CameraGuideVariant
}) {
  const { dashed, label } = getPositionGuideAppearance(state)
  // 밝은 배경에서도 보이도록 어두운 선을 아래에 한 겹 깐다
  const layers = [
    { className: 'stroke-black/50', strokeWidth: 5, strokeDasharray: undefined },
    {
      className: 'stroke-white',
      strokeWidth: 3,
      strokeDasharray: dashed ? '10 8' : undefined,
    },
  ]

  return (
    <div
      aria-hidden="true"
      className={`pointer-events-none absolute inset-x-[4%] bottom-0 z-10 flex items-center justify-center ${
        variant === 'silhouette' ? 'top-[7%]' : 'top-1'
      }`}
    >
      <div className="relative aspect-[540/491] h-full max-w-full">
        <svg
          viewBox={`0 0 ${GUIDE_WIDTH} ${GUIDE_HEIGHT}`}
          className="absolute inset-0 h-full w-full"
          preserveAspectRatio="xMidYMid meet"
        >
          {layers.map((layer) => (
            <g
              key={layer.strokeWidth}
              className={layer.className}
              fill="none"
              strokeWidth={layer.strokeWidth}
              strokeDasharray={layer.strokeDasharray}
              strokeLinecap="round"
            >
              <GuideShape variant={variant} />
            </g>
          ))}
        </svg>

        {/* 손이 안 보일 때만 손을 올릴 위치를 표시한다 */}
        {state.calibrationStatus === 'preparing' && !state.isRecording && !state.isPredicting && state.missingPart === 'hand' && (
          <>
            <div className={`left-[23%] ${HAND_MARKER_CLASS}`}>
              <Hand className="h-8 w-8 text-white sm:h-9 sm:w-9" />
            </div>
            <div className={`right-[23%] ${HAND_MARKER_CLASS}`}>
              <Hand className="h-8 w-8 -scale-x-100 text-white sm:h-9 sm:w-9" />
            </div>
          </>
        )}

        {label && (
          <span className="absolute bottom-[max(26%,72px)] left-1/2 -translate-x-1/2 whitespace-nowrap rounded-full bg-neutral-950/80 px-4 py-2 text-sm font-semibold leading-none text-white sm:text-base">
            {label}
          </span>
        )}
      </div>
    </div>
  )
}
