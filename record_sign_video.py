import cv2
import mediapipe as mp

def record_video(output_path="test_video.mp4", record_seconds=5):
    cap = cv2.VideoCapture(0)
    # 웹캠의 실제 FPS 확인 (보통 30)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 코덱 설정 및 비디오 라이터 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    print(f"🎬 {record_seconds}초 동안 녹화를 시작합니다. 준비하세요!")
    
    with mp_holistic.Holistic(min_detection_confidence=0.5) as holistic:
        for _ in range(int(fps * record_seconds)):
            ret, frame = cap.read()
            if not ret: break
            
            # 녹화는 정방향으로 저장 (flip 안함)
            out.write(frame)
            
            # 화면 표시용 시각화 (사용자 편의)
            display_frame = frame.copy()
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(display_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(display_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            cv2.putText(display_frame, "RECORDING...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Recording...', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 녹화 완료: {output_path}")

if __name__ == "__main__":
    record_video()