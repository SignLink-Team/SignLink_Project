/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Camera, Hand, ArrowDown, VideoOff, Sparkles, Upload, Film, X, Play, Pause } from 'lucide-react';
import { GESTURE_PRESETS } from '../data';
import { GesturePreset } from '../types';
import { useSignSocket } from "../hooks/useSocket";
import { initMediaPipe, startFrameLoop, stopFrameLoop, flattenLandmarks, flattenFaceLandmarks} from "../utils/MediaPipeTracker";

interface CameraViewProps {
  isActive: boolean;
  onToggleActive: () => void;
  onTriggerTranslation: (text: string, category: string) => void;
}

type InputMode = 'camera' | 'video';

export const CameraView: React.FC<CameraViewProps> = ({
  isActive,
  onToggleActive,
  onTriggerTranslation,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

   const {sendFrame, connected } = useSignSocket(
    (data)=>{
      if(data.text){
        onTriggerTranslation(
          data.text,
          "AI"
        );
      }});

  const [inputMode, setInputMode] = useState<InputMode>('camera');
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isUsingRealCamera, setIsUsingRealCamera] = useState<boolean>(false);
  const [activeGesture, setActiveGesture] = useState<GesturePreset | null>(null);
  const [trackingScore, setTrackingScore] = useState<number>(0);

  // Video file state
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoObjectUrl, setVideoObjectUrl] = useState<string | null>(null);
  const [isVideoPaused, setIsVideoPaused] = useState<boolean>(false);
  const [videoDuration, setVideoDuration] = useState<number>(0);
  const [videoCurrentTime, setVideoCurrentTime] = useState<number>(0);
  const [mpReady, setMpReady] = useState(false);


  // ── Camera mode ──────────────────────────────────────────────
  useEffect(() => {
    if (inputMode !== 'camera') return;

    if (isActive) {
      setCameraError(null);
      navigator.mediaDevices
        .getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } })
        .then((stream) => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            streamRef.current = stream;
            setIsUsingRealCamera(true);
          }
        })
        .catch((err) => {
          console.warn('Webcam unavailable, using synthetic tracking:', err);
          setIsUsingRealCamera(false);
          setCameraError(err.message || '카메라를 연결할 수 없습니다.');
        });
    } else {
      stopCameraStream();
    }

    return () => { stopCameraStream(); };
  }, [isActive, inputMode]);

  const stopCameraStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setIsUsingRealCamera(false);
  };

  // ── Video file mode ───────────────────────────────────────────
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Revoke previous object URL to free memory
    if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);

    const url = URL.createObjectURL(file);
    setVideoFile(file);
    setVideoObjectUrl(url);
    setIsVideoPaused(false);

    // Switch to video mode and mark active
    setInputMode('video');
    if (!isActive) onToggleActive();
  };

  // Attach object URL to video element when it changes
  useEffect(() => {
    if (inputMode !== 'video' || !videoRef.current || !videoObjectUrl) return;

    // Clear any existing camera stream first
    stopCameraStream();

    videoRef.current.srcObject = null;
    videoRef.current.src = videoObjectUrl;
    videoRef.current.load();
    videoRef.current.play().catch(() => setIsVideoPaused(true));
  }, [videoObjectUrl, inputMode]);

  // Track video progress
  useEffect(() => {
    const video = videoRef.current;
    if (!video || inputMode !== 'video') return;

    const onTimeUpdate = () => setVideoCurrentTime(video.currentTime);
    const onDurationChange = () => setVideoDuration(video.duration);
    const onPause = () => setIsVideoPaused(true);
    const onPlay = () => setIsVideoPaused(false);

    video.addEventListener('timeupdate', onTimeUpdate);
    video.addEventListener('durationchange', onDurationChange);
    video.addEventListener('pause', onPause);
    video.addEventListener('play', onPlay);

    return () => {
      video.removeEventListener('timeupdate', onTimeUpdate);
      video.removeEventListener('durationchange', onDurationChange);
      video.removeEventListener('pause', onPause);
      video.removeEventListener('play', onPlay);
    };
  }, [inputMode]);

  const handleClearVideo = () => {
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.src = '';
    }
    if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
    setVideoFile(null);
    setVideoObjectUrl(null);
    setIsVideoPaused(false);
    setInputMode('camera');
    if (isActive) onToggleActive();
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const toggleVideoPlayback = () => {
    const video = videoRef.current;
    if (!video) return;
    if (isVideoPaused) {
      video.play().catch(console.error);
    } else {
      video.pause();
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = Number(e.target.value);
  };

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  // ── Canvas skeleton overlay (camera mode only) ────────────────
  useEffect(() => {
    if (!isActive || inputMode !== 'camera') return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationId: number;
    let frame = 0;

    const render = () => {
      frame++;
      const width = (canvas.width = canvas.parentElement?.clientWidth || 640);
      const height = (canvas.height = canvas.parentElement?.clientHeight || 400);
      ctx.clearRect(0, 0, width, height);

      const scanY = (frame * 3) % height;
      ctx.strokeStyle = 'rgba(15, 194, 158, 0.15)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, scanY);
      ctx.lineTo(width, scanY);
      ctx.stroke();

      ctx.fillStyle = 'rgba(15, 194, 158, 0.7)';
      ctx.font = '10px monospace';
      ctx.fillText(`FPS: 60 | ID: SIGN_ENGINE_V4 | LATENCY: 12ms`, 16, height - 20);

      setTrackingScore(Math.round(88 + Math.sin(frame / 20) * 11));

      const centerX = width / 2;
      const centerY = height / 2 + Math.sin(frame / 15) * 15;
      const wrist = { x: centerX, y: centerY + 80 };
      const fingers = [
        { offset: -60, lengths: [30, 20, 15], angle: -0.6 },
        { offset: -30, lengths: [40, 25, 20], angle: -0.2 },
        { offset: 0,   lengths: [48, 28, 22], angle: 0    },
        { offset: 30,  lengths: [42, 26, 20], angle: 0.15 },
        { offset: 60,  lengths: [32, 20, 16], angle: 0.4  },
      ];

      fingers.forEach((fin) => {
        const baseK = {
          x: centerX + fin.offset + Math.sin(frame / 30) * 10,
          y: centerY + Math.cos(fin.offset / 50) * 10,
        };
        ctx.strokeStyle = 'rgba(15, 194, 158, 0.3)';
        ctx.beginPath();
        ctx.moveTo(wrist.x, wrist.y);
        ctx.lineTo(baseK.x, baseK.y);
        ctx.stroke();

        let curX = baseK.x;
        let curY = baseK.y;
        ctx.strokeStyle = '#0fc29e';
        ctx.beginPath();
        ctx.moveTo(curX, curY);

        fin.lengths.forEach((len, idx) => {
          const bend = Math.sin(frame / 10 + idx + fin.offset) * 0.15;
          const angle = fin.angle + bend;
          const nextX = curX + Math.sin(angle) * len;
          const nextY = curY - Math.cos(angle) * len;
          ctx.lineTo(nextX, nextY);
          ctx.fillStyle = idx === fin.lengths.length - 1 ? '#ffffff' : '#0fc29e';
          ctx.beginPath();
          ctx.arc(nextX, nextY, idx === fin.lengths.length - 1 ? 4 : 3, 0, Math.PI * 2);
          ctx.fill();
          curX = nextX;
          curY = nextY;
        });
        ctx.strokeStyle = 'rgba(15, 194, 158, 0.7)';
        ctx.stroke();
      });

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(centerX, centerY + 20, 110 + Math.sin(frame / 10) * 8, 0, Math.PI * 2);
      ctx.stroke();

      animationId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animationId);
  }, [isActive, inputMode]);

  // ── Gesture preset handler ────────────────────────────────────
  const handlePresetSelect = (preset: GesturePreset) => {
    setActiveGesture(preset);
    onTriggerTranslation(preset.translationText, preset.category);
    setTimeout(() => setActiveGesture(null), 1500);
  };

  const isVideoMode = inputMode === 'video' && !!videoFile;
  useEffect(() => {
    initMediaPipe().then(() => {
    console.log("MediaPipe 초기화 완료");
    setMpReady(true);
    });
  }, []);

// ── 프레임 루프 시작/종료 ─────────────────────────
  useEffect(() => {
    if (!isActive || !videoRef.current || !mpReady || !connected) return;
    console.log("프레임 루프 시작 - 소켓 연결 확인");
    startFrameLoop(videoRef.current, (result) => {
      console.log("MediaPipe result:", result);
      sendFrame({
        frame_id: Date.now(),
        수지: flattenLandmarks(result),
        비수지: flattenFaceLandmarks(result),
      });
    });

  return () => stopFrameLoop();
}, [isActive, videoObjectUrl, mpReady, connected]);

  // ── Render ────────────────────────────────────────────────────
  return (
    <div className="w-full">
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={handleFileChange}
      />

      {/* ── Mode selector ── */}
      {!isActive && (
        <div className="flex items-center gap-2 mb-3">
          <button
            onClick={() => { setInputMode('camera'); }}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-sm font-bold border transition-all ${
              inputMode === 'camera'
                ? 'bg-brand-green/10 border-brand-green text-brand-green'
                : 'bg-white border-neutral-200 text-neutral-500 hover:border-neutral-300'
            }`}
          >
            <Camera className="w-4 h-4" />
            실시간 카메라
          </button>

          <button
            onClick={() => fileInputRef.current?.click()}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-sm font-bold border transition-all ${
              inputMode === 'video'
                ? 'bg-brand-green/10 border-brand-green text-brand-green'
                : 'bg-white border-neutral-200 text-neutral-500 hover:border-neutral-300'
            }`}
          >
            <Upload className="w-4 h-4" />
            동영상 업로드
          </button>
        </div>
      )}

      {/* ── Viewport ── */}
      <div
        className="relative w-full aspect-video min-h-[280px] sm:min-h-[420px] bg-brand-emerald rounded-2xl flex flex-col items-center justify-center overflow-hidden border border-emerald-950/20 shadow-inner group cursor-pointer"
        onClick={() => { if (!isActive && !isVideoMode) onToggleActive(); }}
      >
        {/* Video element — shared for both camera and file */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-300 ${
            isActive || isVideoMode
              ? isVideoMode ? 'opacity-100' : 'opacity-75 mix-blend-screen'
              : 'hidden'
          }`}
        />

        {/* Canvas overlay (camera skeleton — camera mode only) */}
        {isActive && inputMode === 'camera' && (
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none z-10"
          />
        )}

        {/* Ambient gradients */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/50 via-transparent to-black/10 pointer-events-none z-10" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(15,194,158,0.1)_0%,rgba(0,0,0,0.4)_100%)] pointer-events-none z-10" />

        <AnimatePresence mode="wait">
          {/* ── Inactive idle screen ── */}
          {!isActive && !isVideoMode && (
            <motion.div
              key="idle"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="flex flex-col items-center justify-center p-6 text-center select-none z-20"
            >
              <motion.div
                whileHover={{ scale: 1.08 }}
                className="w-20 h-20 rounded-full bg-white/10 hover:bg-white/15 text-white flex items-center justify-center border border-white/20 shadow-lg cursor-pointer transition-all duration-300"
              >
                <Hand className="w-10 h-10 text-white transform -rotate-12 animate-bounce" />
              </motion.div>

              <div className="mt-6 flex flex-col items-center gap-1.5">
                <div className="w-10 h-10 rounded-full bg-white/5 border border-white/10 flex items-center justify-center text-white/90">
                  <Camera className="w-5 h-5 animate-pulse" />
                </div>
                <ArrowDown className="w-5 h-5 text-brand-green mt-1" />
              </div>

              <h3 className="mt-5 text-xl font-bold text-white tracking-wide">
                화면을 눌러 카메라 실행
              </h3>
              <p className="text-sm text-neutral-300 max-w-sm mt-2 font-medium">
                또는 위에서 동영상을 업로드해 번역을 테스트하세요.
              </p>
            </motion.div>
          )}

          {/* ── Camera active HUD ── */}
          {isActive && inputMode === 'camera' && (
            <motion.div
              key="camera-hud"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute top-4 left-4 right-4 flex items-center justify-between z-20 pointer-events-none"
            >
              <div className="flex items-center gap-2 bg-black/50 backdrop-blur-md px-3.5 py-1.5 rounded-full border border-white/10">
                <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-ping" />
                <span className="text-xs font-bold text-white tracking-wider ml-0.5">LIVE 수어 감지 중</span>
              </div>
              <div className="flex items-center gap-2 bg-black/50 backdrop-blur-md px-3.5 py-1.5 rounded-full border border-white/10">
                <Sparkles className="w-4 h-4 text-brand-green animate-pulse" />
                <span className="text-xs font-bold text-white tracking-wider">정합도: {trackingScore}%</span>
              </div>
            </motion.div>
          )}

          {/* ── Video file HUD ── */}
          {isVideoMode && (
            <motion.div
              key="video-hud"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute top-4 left-4 right-4 flex items-center justify-between z-20 pointer-events-none"
            >
              <div className="flex items-center gap-2 bg-black/50 backdrop-blur-md px-3.5 py-1.5 rounded-full border border-white/10 max-w-[60%]">
                <Film className="w-3.5 h-3.5 text-brand-green shrink-0" />
                <span className="text-xs font-bold text-white truncate">{videoFile.name}</span>
              </div>
              <div className="flex items-center gap-2 bg-black/50 backdrop-blur-md px-3.5 py-1.5 rounded-full border border-white/10">
                <span className="text-xs font-bold text-white tabular-nums">
                  {formatTime(videoCurrentTime)} / {formatTime(videoDuration)}
                </span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Camera off button ── */}
        {isActive && inputMode === 'camera' && (
          <div className="absolute bottom-4 right-4 z-20">
            <button
              onClick={(e) => { e.stopPropagation(); onToggleActive(); }}
              className="px-3.5 py-2 bg-black/60 hover:bg-black/80 backdrop-blur-md text-white rounded-lg border border-white/10 text-sm font-semibold flex items-center gap-1.5 transition-colors"
            >
              <VideoOff className="w-4 h-4 text-red-400" />
              카메라 끄기
            </button>
          </div>
        )}

        {/* ── Video controls overlay ── */}
        {isVideoMode && (
          <div className="absolute bottom-0 left-0 right-0 z-20 bg-gradient-to-t from-black/80 to-transparent px-4 pb-4 pt-8">
            {/* Seek bar */}
            <input
              type="range"
              min={0}
              max={videoDuration || 0}
              step={0.1}
              value={videoCurrentTime}
              onChange={handleSeek}
              onClick={(e) => e.stopPropagation()}
              className="w-full h-1 accent-brand-green cursor-pointer mb-3"
            />
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {/* Play/Pause */}
                <button
                  onClick={(e) => { e.stopPropagation(); toggleVideoPlayback(); }}
                  className="w-8 h-8 rounded-full bg-white/15 hover:bg-white/25 flex items-center justify-center transition-colors"
                >
                  {isVideoPaused
                    ? <Play className="w-4 h-4 text-white ml-0.5" />
                    : <Pause className="w-4 h-4 text-white" />}
                </button>

                {/* Upload new file */}
                <button
                  onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-white/10 hover:bg-white/20 text-white rounded-lg text-xs font-bold transition-colors"
                >
                  <Upload className="w-3.5 h-3.5" />
                  다른 영상
                </button>
              </div>

              {/* Remove video */}
              <button
                onClick={(e) => { e.stopPropagation(); handleClearVideo(); }}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-red-500/20 hover:bg-red-500/30 text-red-300 rounded-lg text-xs font-bold transition-colors"
              >
                <X className="w-3.5 h-3.5" />
                제거
              </button>
            </div>
          </div>
        )}
      </div>

      {/* ── Simulator pills (camera mode only) ── */}
      {isActive && inputMode === 'camera' && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 bg-white border border-neutral-100 rounded-xl p-4 shadow-xs"
        >
          <div className="flex items-center justify-between border-b pb-2 mb-3">
            <span className="text-sm font-bold text-on-surface flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-secondary" />
              임상 수어 시뮬레이터 (진료 상황 맞춤 테스트)
            </span>
            <span className="text-xs text-on-surface-variant font-medium bg-neutral-100 px-2 py-0.5 rounded-sm">
              제스처 선택 시 즉시 번역
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {GESTURE_PRESETS.map((preset) => {
              const active = activeGesture?.id === preset.id;
              return (
                <button
                  key={preset.id}
                  onClick={() => handlePresetSelect(preset)}
                  className={`p-2 rounded-lg border text-left flex flex-col transition-all duration-200 ${
                    active
                      ? 'border-brand-green bg-brand-green/10 ring-1 ring-brand-green'
                      : 'border-neutral-200 bg-neutral-50 hover:bg-neutral-100 hover:border-neutral-300'
                  }`}
                >
                  <span className="text-xs font-bold text-on-surface truncate">{preset.category} 수어</span>
                  <span className="text-xs text-neutral-500 line-clamp-1 mt-0.5 font-normal">{preset.gestureName}</span>
                </button>
              );
            })}
          </div>
        </motion.div>
      )}

      {/* ── Video mode: simulator pills still available for manual testing ── */}
      {isVideoMode && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 bg-white border border-neutral-100 rounded-xl p-4 shadow-xs"
        >
          <div className="flex items-center justify-between border-b pb-2 mb-3">
            <span className="text-sm font-bold text-on-surface flex items-center gap-1.5">
              <Film className="w-4 h-4 text-secondary" />
              동영상 테스트 — 수동 번역 트리거
            </span>
            <span className="text-xs text-neutral-400 font-medium bg-neutral-100 px-2 py-0.5 rounded-sm">
              MediaPipe 연결 전 임시
            </span>
          </div>
          <p className="text-xs text-neutral-400 mb-3">
            백엔드 WebSocket 연결 후 동영상 프레임에서 자동 번역됩니다. 지금은 아래에서 수동으로 결과를 트리거하세요.
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {GESTURE_PRESETS.map((preset) => {
              const active = activeGesture?.id === preset.id;
              return (
                <button
                  key={preset.id}
                  onClick={() => handlePresetSelect(preset)}
                  className={`p-2 rounded-lg border text-left flex flex-col transition-all duration-200 ${
                    active
                      ? 'border-brand-green bg-brand-green/10 ring-1 ring-brand-green'
                      : 'border-neutral-200 bg-neutral-50 hover:bg-neutral-100 hover:border-neutral-300'
                  }`}
                >
                  <span className="text-xs font-bold text-on-surface truncate">{preset.category} 수어</span>
                  <span className="text-xs text-neutral-500 line-clamp-1 mt-0.5 font-normal">{preset.gestureName}</span>
                </button>
              );
            })}
          </div>
        </motion.div>
      )}
    </div>
  );
};
