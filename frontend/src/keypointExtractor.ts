type Landmark = { x: number; y: number; z: number }

type HolisticResults = {
  poseLandmarks?: Landmark[]
  leftHandLandmarks?: Landmark[]
  rightHandLandmarks?: Landmark[]
}

declare global {
  interface Window {
    Holistic?: new (options: { locateFile: (file: string) => string }) => {
      setOptions: (options: Record<string, unknown>) => void
      onResults: (callback: (results: HolisticResults) => void) => void
      send: (input: { image: HTMLVideoElement | HTMLCanvasElement }) => Promise<void>
      close?: () => Promise<void>
    }
  }
}

let holisticScriptPromise: Promise<void> | null = null

function loadScript(src: string): Promise<void> {
  const existing = document.querySelector(`script[src="${src}"]`)
  if (existing) return Promise.resolve()

  return new Promise((resolve, reject) => {
    const script = document.createElement('script')
    script.src = src
    script.async = true
    script.onload = () => resolve()
    script.onerror = () => reject(new Error('MediaPipe Holistic 스크립트를 불러오지 못했습니다.'))
    document.head.appendChild(script)
  })
}

async function loadHolisticScript() {
  if (!holisticScriptPromise) {
    holisticScriptPromise = loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js')
  }
  await holisticScriptPromise
  if (!window.Holistic) throw new Error('MediaPipe Holistic 초기화에 실패했습니다.')
}

function getAngle(v1: number[], v2: number[]) {
  const norm1 = Math.hypot(...v1)
  const norm2 = Math.hypot(...v2)
  if (norm1 === 0 || norm2 === 0) return 0
  const dot = v1.reduce((sum, value, index) => sum + (value / norm1) * (v2[index] / norm2), 0)
  return Math.acos(Math.max(-1, Math.min(1, dot)))
}

function zeros(length: number) {
  return Array.from({ length }, () => 0)
}

function processHand(handLandmarks: Landmark[] | undefined, noseCoords: number[]) {
  if (!handLandmarks?.length) return { data: zeros(78), rel: zeros(3) }

  const wrist = handLandmarks[0]
  const middleMcp = handLandmarks[9]
  const rel = [wrist.x - noseCoords[0], wrist.y - noseCoords[1], wrist.z - noseCoords[2]]
  const handLength = Math.hypot(middleMcp.x - wrist.x, middleMcp.y - wrist.y, middleMcp.z - wrist.z)
  const handScale = handLength > 0.01 ? handLength : 1
  const coords = handLandmarks.map((point) => [
    (point.x - wrist.x) / handScale,
    (point.y - wrist.y) / handScale,
    (point.z - wrist.z) / handScale,
  ])

  const jointIndices = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
  ]
  const angles: number[] = []
  jointIndices.forEach((finger) => {
    for (let i = 0; i < finger.length - 2; i += 1) {
      const a = coords[finger[i]]
      const b = coords[finger[i + 1]]
      const c = coords[finger[i + 2]]
      angles.push(getAngle([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - b[0], c[1] - b[1], c[2] - b[2]]))
    }
  })

  return { data: coords.flat().concat(angles), rel }
}

function extractNormalizedKeypoints(results: HolisticResults) {
  let noseCoords = [0, 0, 0]
  let poseData = zeros(33 * 3)

  if (results.poseLandmarks?.length) {
    const pose = results.poseLandmarks
    const nose = pose[0]
    noseCoords = [nose.x, nose.y, nose.z]
    const shoulderL = pose[11]
    const shoulderR = pose[12]
    const shoulderWidth = Math.hypot(shoulderL.x - shoulderR.x, shoulderL.y - shoulderR.y, shoulderL.z - shoulderR.z)
    const poseScale = shoulderWidth > 0.01 ? shoulderWidth : 1
    poseData = pose.flatMap((point) => [
      (point.x - nose.x) / poseScale,
      (point.y - nose.y) / poseScale,
      (point.z - nose.z) / poseScale,
    ])
  }

  const leftHand = processHand(results.leftHandLandmarks, noseCoords)
  const rightHand = processHand(results.rightHandLandmarks, noseCoords)
  return poseData.concat(leftHand.data, rightHand.data, leftHand.rel, rightHand.rel)
}

export class BrowserKeypointExtractor {
  private holistic: InstanceType<NonNullable<typeof window.Holistic>> | null = null
  private latest: number[] | null = null

  async init() {
    await loadHolisticScript()
    this.holistic = new window.Holistic!({
      locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    })
    this.holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      refineFaceLandmarks: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    })
    this.holistic.onResults((results) => {
      this.latest = extractNormalizedKeypoints(results)
    })
  }

  async extract(video: HTMLVideoElement) {
    if (!this.holistic) await this.init()
    this.latest = null
    await this.holistic!.send({ image: video })
    return this.latest
  }

  async close() {
    await this.holistic?.close?.()
    this.holistic = null
  }
}
