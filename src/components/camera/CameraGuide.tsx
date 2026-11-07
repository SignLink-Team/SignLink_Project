import { Fragment } from 'react'
import { ChevronRight, Hand } from 'lucide-react'

// CameraView의 사용 안내(단계 표시, 사용 방법 카드, 상반신 구도 가이드)와 안내 문구.

// done: 한 문장의 수어를 마친 상태. '다시 맞추기'를 누르기 전까지 새 수어 인식을 시작하지 않는다.
export type MotionCalibrationStatus = 'preparing' | 'calibrating' | 'unstable' | 'ready' | 'done'
/** 보정이 시작되지 못하는 이유(상반신/손 미검출) */
export type MissingPart = 'body' | 'hand' | null

interface GuideText {
  title: string
  description: string
}

// 상반신(머리~가슴) 구도 가이드 실루엣.
// 업로드한 가이드 이미지에서 윤곽만 벡터로 옮긴 것이다(모서리 꺾쇠와 배경은 제외).
// viewBox는 width x height, path는 머리/목과 어깨 두 덩어리로 이루어져 있다.
const UPPER_BODY_GUIDE = {
  width: 402,
  height: 491,
  path:
    'M3.0 406.0 L3.0 430.0 L4.0 431.0 L4.0 435.0 L21.0 447.0 L40.0 457.0 L78.0 471.0 L81.0 471.0 L102.0 477.0 L106.0 477.0 L107.0 478.0 L111.0 478.0 L117.0 480.0 L122.0 480.0 L123.0 481.0 L128.0 481.0 L129.0 482.0 L136.0 482.0 L137.0 483.0 L143.0 483.0 L144.0 484.0 L153.0 484.0 L154.0 485.0 L166.0 485.0 L167.0 486.0 L213.0 487.0 L214.0 486.0 L247.0 485.0 L248.0 484.0 L257.0 484.0 L258.0 483.0 L264.0 483.0 L265.0 482.0 L272.0 482.0 L273.0 481.0 L284.0 480.0 L285.0 479.0 L299.0 477.0 L304.0 475.0 L308.0 475.0 L344.0 464.0 L359.0 457.0 L361.0 457.0 L375.0 450.0 L397.0 435.0 L397.0 431.0 L398.0 430.0 L398.0 406.0 L397.0 405.0 L397.0 396.0 L396.0 395.0 L396.0 389.0 L395.0 388.0 L392.0 371.0 L386.0 354.0 L377.0 336.0 L366.0 320.0 L341.0 294.0 L330.0 286.0 L313.0 277.0 L311.0 277.0 L306.0 274.0 L291.0 269.0 L288.0 267.0 L283.0 266.0 L267.0 258.0 L264.0 258.0 L259.0 265.0 L239.0 284.0 L224.0 292.0 L214.0 295.0 L209.0 295.0 L208.0 296.0 L193.0 296.0 L192.0 295.0 L187.0 295.0 L177.0 292.0 L167.0 287.0 L158.0 281.0 L142.0 265.0 L137.0 258.0 L134.0 258.0 L125.0 263.0 L95.0 274.0 L76.0 283.0 L60.0 294.0 L39.0 315.0 L24.0 336.0 L15.0 354.0 L9.0 371.0 L9.0 374.0 L6.0 383.0 L5.0 395.0 L4.0 396.0 L4.0 405.0 Z M187.0 3.0 L186.0 4.0 L180.0 4.0 L179.0 5.0 L170.0 6.0 L156.0 11.0 L146.0 16.0 L136.0 23.0 L120.0 40.0 L115.0 48.0 L108.0 64.0 L108.0 67.0 L105.0 76.0 L105.0 81.0 L104.0 82.0 L103.0 112.0 L100.0 114.0 L95.0 120.0 L95.0 123.0 L94.0 124.0 L94.0 135.0 L95.0 136.0 L97.0 146.0 L101.0 154.0 L107.0 161.0 L109.0 171.0 L114.0 186.0 L125.0 207.0 L131.0 215.0 L143.0 227.0 L159.0 238.0 L159.0 259.0 L162.0 262.0 L177.0 270.0 L190.0 274.0 L199.0 274.0 L201.0 275.0 L202.0 274.0 L211.0 274.0 L224.0 270.0 L239.0 262.0 L242.0 259.0 L242.0 238.0 L258.0 227.0 L270.0 215.0 L279.0 202.0 L287.0 186.0 L292.0 171.0 L294.0 161.0 L298.0 157.0 L304.0 146.0 L306.0 139.0 L306.0 135.0 L307.0 134.0 L307.0 124.0 L305.0 118.0 L298.0 112.0 L297.0 82.0 L296.0 81.0 L296.0 76.0 L295.0 75.0 L293.0 64.0 L286.0 48.0 L278.0 36.0 L261.0 20.0 L255.0 16.0 L240.0 9.0 L231.0 6.0 L227.0 6.0 L221.0 4.0 L215.0 4.0 L214.0 3.0 Z',
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

const STATUS_LABEL: Record<MotionCalibrationStatus, string> = {
  preparing: '자리 맞추기',
  calibrating: '가만히 있기',
  unstable: '다시 맞추기 필요',
  ready: '수어 시작 가능',
  done: '번역 완료',
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

/** 영상 왼쪽 위 HUD의 상태 문구 */
export function getStatusText(state: GuideState): string {
  const currentStep = getCurrentStep(state)
  if (state.isRecording) return `${currentStep}/4 수어 인식 중`
  if (state.isPredicting) return '번역 중'
  if (state.calibrationStatus === 'done') return STATUS_LABEL.done
  return `${currentStep}/4 ${STATUS_LABEL[state.calibrationStatus]}`
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
export function GuideSteps({ current }: { current: number }) {
  // 모든 단계를 마치면(번역 중/번역 완료) 진행 방향 화살표는 숨긴다.
  const allDone = current > GUIDE_STEPS.length

  return (
    <ol aria-label="사용 단계" className="flex items-start gap-1 sm:gap-2">
      {GUIDE_STEPS.map((step, index) => {
        const number = index + 1
        const done = number < current
        const active = number === current
        const showArrow = !allDone && number < GUIDE_STEPS.length

        return (
          <Fragment key={step.title}>
            <li
              aria-current={active ? 'step' : undefined}
              className="flex min-w-0 flex-1 flex-col items-center gap-1.5 text-center"
            >
              {/* 지금 단계는 불이 들어온 것처럼 밝게 빛나고, 지난 단계는 초록색으로 켜진 채 남는다. */}
              <span
                className={`flex h-9 w-9 items-center justify-center rounded-full text-base font-bold transition-all duration-500 motion-reduce:transition-none sm:h-10 sm:w-10 sm:text-lg ${
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
                className={`break-keep text-sm leading-snug transition-colors duration-500 motion-reduce:transition-none sm:text-base ${
                  active ? 'font-bold text-white' : done ? 'font-medium text-emerald-200' : 'text-white/60'
                }`}
              >
                {step.title}
              </span>
            </li>

            {/* 단계 사이 화살표. 앞 단계를 마치면 초록색으로 켜진다. 원의 가운데 높이에 맞춘다. */}
            {showArrow && (
              <li aria-hidden="true" className="mt-2.5 shrink-0 sm:mt-3">
                <ChevronRight
                  className={`h-5 w-5 transition-colors duration-500 motion-reduce:transition-none ${
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
  'absolute top-[60%] flex h-14 w-14 animate-pulse items-center justify-center rounded-2xl border-2 border-dashed border-white/90 bg-neutral-950/40 motion-reduce:animate-none sm:h-16 sm:w-16'

interface PositionGuideOverlayProps {
  missingPart: MissingPart
  /**
   * 보정 단계가 아닐 때(수어 시작 가능/인식 중/번역 중/번역 완료)는 윤곽만 표시하고,
   * 수어 동작을 가리지 않도록 안내 문구와 손 위치 표시는 숨긴다.
   */
  outlineOnly?: boolean
}

/** 구도 안내 전용: 상반신(머리~가슴)이 윤곽 안에 들어오도록 거리를 맞춘다. 실제 검출 판정은 하지 않는다. */
export function PositionGuideOverlay({ missingPart, outlineOnly = false }: PositionGuideOverlayProps) {
  // 영상 위 실루엣에 붙는 한 줄 안내
  const guideLabel = missingPart === 'hand' ? '두 손을 가슴 앞으로 올려 주세요' : '윤곽에 상반신을 맞춰 주세요'

  return (
    <div
      aria-hidden="true"
      className="pointer-events-none absolute inset-x-[8%] bottom-[5%] top-[13%] z-10 flex items-center justify-center"
    >
      <div className="relative aspect-[402/491] h-full max-w-full">
        <svg
          viewBox={`0 0 ${UPPER_BODY_GUIDE.width} ${UPPER_BODY_GUIDE.height}`}
          className="absolute inset-0 h-full w-full"
          preserveAspectRatio="xMidYMid meet"
        >
          {/* 밝은 배경에서도 보이도록 어두운 테두리를 한 겹 깐다 */}
          <path
            d={UPPER_BODY_GUIDE.path}
            fill="none"
            stroke="rgba(0, 0, 0, 0.5)"
            strokeWidth={5}
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />
          <path
            d={UPPER_BODY_GUIDE.path}
            fill="rgba(255, 255, 255, 0.16)"
            stroke="rgba(255, 255, 255, 0.92)"
            strokeWidth={2}
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />
        </svg>

        {/* 손이 안 보일 때만 손을 올릴 위치를 표시한다 */}
        {!outlineOnly && missingPart === 'hand' && (
          <>
            <div className={`left-[14%] ${HAND_MARKER_CLASS}`}>
              <Hand className="h-8 w-8 text-white sm:h-9 sm:w-9" />
            </div>
            <div className={`right-[14%] ${HAND_MARKER_CLASS}`}>
              <Hand className="h-8 w-8 -scale-x-100 text-white sm:h-9 sm:w-9" />
            </div>
          </>
        )}

        {!outlineOnly && (
          <span className="absolute bottom-[6%] left-1/2 -translate-x-1/2 whitespace-nowrap rounded-full bg-neutral-950/80 px-4 py-2 text-sm font-semibold leading-none text-white sm:text-base">
            {guideLabel}
          </span>
        )}
      </div>
    </div>
  )
}
