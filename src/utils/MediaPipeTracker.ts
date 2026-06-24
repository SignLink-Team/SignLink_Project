import {
  FilesetResolver,
  HolisticLandmarker,
  HolisticLandmarkerResult
} from "@mediapipe/tasks-vision";

let landmarker: HolisticLandmarker | null = null;
let animationId: number | null = null;

export async function initMediaPipe() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm" 
  );

  landmarker = await HolisticLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task",  // ← 모델도 CDN
      delegate: "GPU"
    },
    runningMode: "VIDEO"
  });
}

function getAngle(v1: number[], v2: number[]): number {
  const dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
  const n1 = Math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2);
  const n2 = Math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2);
  if (n1 < 1e-6 || n2 < 1e-6) return 0;
  return Math.acos(Math.max(-1, Math.min(1, dot / (n1 * n2))));
}

export function flattenLandmarks(result: HolisticLandmarkerResult): number[] {
  const features: number[] = [];

  // ── 포즈 (99차원) ──────────────────────────────
  const pose = result.poseLandmarks?.[0];
  let noseCoords = [0, 0, 0];

  if (pose && pose.length > 0) {
    const nose = pose[0];
    noseCoords = [nose.x, nose.y, nose.z];

    const sl = pose[11], sr = pose[12];
    const shoulderWidth = Math.sqrt(
      (sl.x - sr.x)**2 + (sl.y - sr.y)**2 + (sl.z - sr.z)**2
    );
    const scale = shoulderWidth > 0.01 ? shoulderWidth : 1.0;

    pose.forEach(p => {
      features.push(
        (p.x - nose.x) / scale,
        (p.y - nose.y) / scale,
        (p.z - nose.z) / scale,
      );
    });
  } else {
    features.push(...new Array(99).fill(0));
  }

  // ── 손 처리 공통 함수 (78차원) ─────────────────
  const processHand = (
    landmarks: {x:number, y:number, z:number}[] | undefined
  ): { data: number[]; rel: number[] } => {
    if (!landmarks || landmarks.length === 0) {
      return { data: new Array(78).fill(0), rel: [0, 0, 0] };
    }

    const wrist = landmarks[0];
    const rel = [
      wrist.x - noseCoords[0],
      wrist.y - noseCoords[1],
      wrist.z - noseCoords[2],
    ];

    const mcp = landmarks[9];
    const handLength = Math.sqrt(
      (mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2 + (mcp.z - wrist.z)**2
    );
    const scale = handLength > 0.01 ? handLength : 1.0;

    // 정규화 좌표 (63차원)
    const coords = landmarks.map(p => ([
      (p.x - wrist.x) / scale,
      (p.y - wrist.y) / scale,
      (p.z - wrist.z) / scale,
    ]));

    // 관절 각도 (15차원)
    const fingerIndices = [
      [0,1,2,3,4], [0,5,6,7,8],
      [0,9,10,11,12], [0,13,14,15,16], [0,17,18,19,20],
    ];
    const angles: number[] = [];
    fingerIndices.forEach(finger => {
      for (let i = 0; i < finger.length - 2; i++) {
        const v1 = [
          coords[finger[i+1]][0] - coords[finger[i]][0],
          coords[finger[i+1]][1] - coords[finger[i]][1],
          coords[finger[i+1]][2] - coords[finger[i]][2],
        ];
        const v2 = [
          coords[finger[i+2]][0] - coords[finger[i+1]][0],
          coords[finger[i+2]][1] - coords[finger[i+1]][1],
          coords[finger[i+2]][2] - coords[finger[i+1]][2],
        ];
        angles.push(getAngle(v1, v2));
      }
    });

    return { data: [...coords.flat(), ...angles], rel };
  };

  const lh = processHand(result.leftHandLandmarks?.[0]);
  const rh = processHand(result.rightHandLandmarks?.[0]);

  features.push(...lh.data);   // 78
  features.push(...rh.data);   // 78
  features.push(...lh.rel);    // 3
  features.push(...rh.rel);    // 3

  // 총 261차원 — velocity/acceleration은 백엔드에서 추가
  return features;
}

export function flattenFaceLandmarks(result: HolisticLandmarkerResult): number[] {
  const face = result.faceLandmarks?.[0];
  if (!face || face.length === 0) return new Array(468 * 3).fill(0);

  const features: number[] = [];
  face.forEach(p => features.push(p.x, p.y, p.z));
  return features;
}


// 추가 필요: 프레임 루프
export function startFrameLoop(
  video: HTMLVideoElement,
  onResult: (result: HolisticLandmarkerResult) => void
) {
  if (!landmarker) return;

  const loop = () => {
    // 비디오 준비 안 됐으면 스킵하고 다음 프레임에 재시도
    if (
      video.readyState < 2 ||        // HAVE_CURRENT_DATA 미만
      video.videoWidth === 0 ||       // 아직 크기 없음
      video.videoHeight === 0 ||
      video.paused ||
      video.ended
    ) {
      animationId = requestAnimationFrame(loop);
      return;
    }

    try {
      const result = landmarker!.detectForVideo(video, performance.now());
      onResult(result);
    } catch (e) {
      console.warn("MediaPipe frame skip:", e);
    }

    animationId = requestAnimationFrame(loop);
  };

  animationId = requestAnimationFrame(loop);
}

export function stopFrameLoop() {
  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
}