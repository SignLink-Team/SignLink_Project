import cv2
import numpy as np
import os
import json
import mediapipe as mp
import concurrent.futures
from scipy.ndimage import gaussian_filter1d
from nonmanual_features import extract_nonmanual, NMS_KEYS, FEATURE_DIM

BASE_DIR = r"D:\새 폴더 (2)"
TARGET_DIRS = [
    "02_NIKL_Sign Language Parallel Corpus_2023_Me1",
    "02_NIKL_Sign Language Parallel Corpus_2023_Me2",
    "02_NIKL_Sign Language Parallel Corpus_2023_Me3",
]
SAVE_PATH           = "./keypoint_data"
NONMANUAL_SAVE_PATH = os.path.join(SAVE_PATH, "nonmanual_features")
MANUAL_INFO_PATH    = os.path.join(SAVE_PATH, "nms_dataset_info.json")

os.makedirs(NONMANUAL_SAVE_PATH, exist_ok=True)

# 영벡터 비율 허용 상한 (이 이상이면 샘플 제외)
ZERO_RATIO_THRESHOLD = 0.3


def build_label_index():
    index = {}
    for folder in TARGET_DIRS:
        folder_path = os.path.join(BASE_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"⚠️  폴더 없음: {folder_path}")
            continue
        for root, _, files in os.walk(folder_path):
            for fname in files:
                if not fname.endswith('.json'):
                    continue
                label_path = os.path.join(root, fname)
                try:
                    with open(label_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    sample_id      = data.get('id')
                    video_filename = data.get('vido_file_nm', '') + ".mp4"
                    video_path     = os.path.join(root, video_filename)
                    if sample_id and os.path.exists(video_path):
                        index[sample_id] = (label_path, video_path)
                except Exception:
                    continue
    print(f"인덱스 구축 완료: {len(index)}개 파일")
    return index


def build_nms_labels(nms_script, total_frames, fps, start_frame):
    nms_labels = {key: [] for key in NMS_KEYS}
    for raw_key, segments in nms_script.items():
        key = next((k for k in NMS_KEYS if k.lower() == raw_key.lower()), None)
        if key is None or not segments:
            continue
        for seg in segments:
            seg_start = int(seg["start"] * fps) - start_frame
            seg_end   = int(seg["end"]   * fps) - start_frame
            seg_start = max(0, seg_start)
            seg_end   = min(total_frames - 1, seg_end)
            if seg_start <= seg_end:
                nms_labels[key].append({
                    "start_frame": seg_start,
                    "end_frame":   seg_end,
                })
    return nms_labels


def process_single_file(args):
    label_path, video_path = args

    with open(label_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    gestures = data.get('sign_script', {}).get('sign_gestures_strong', [])
    if not gestures:
        return None

    nms_script = data.get('nms_script', {})
    fps        = data.get('potogrf', {}).get('fps', 30)

    start_time    = min(item['start'] for item in gestures)
    end_time      = max(item['end']   for item in gestures)
    margin_frames = int(fps * 0.5)
    start_frame   = max(0, int(start_time * fps) - margin_frames)
    end_frame     = int(end_time * fps) + margin_frames

    nonmanual_seq  = []
    prev_landmarks = None

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as face_mesh:
        while cap.isOpened() and current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frame   = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])),
                                  interpolation=cv2.INTER_LINEAR)
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                prev_landmarks = face_landmarks
            else:
                face_landmarks = prev_landmarks  # 직전 프레임으로 대체

            nonmanual_seq.append(extract_nonmanual(face_landmarks, frame.shape))
            current_frame += 1
    cap.release()

    if len(nonmanual_seq) < 5:
        return None

    features = gaussian_filter1d(np.array(nonmanual_seq), sigma=1.0, axis=0)
    # vel/acc 없음 — 좌표만 사용

    # 영벡터 비율 체크 → 30% 초과 시 제외
    zero_ratio = (features.sum(axis=1) == 0).sum() / len(features)
    if zero_ratio > ZERO_RATIO_THRESHOLD:
        return None

    nms_labels     = build_nms_labels(nms_script, len(features), fps, start_frame)
    nonmanual_file = f"{data['id']}_nonmanual.npy"
    np.save(os.path.join(NONMANUAL_SAVE_PATH, nonmanual_file), features)

    return {
        "id":                     data['id'],
        "nonmanual_feature_file": nonmanual_file,
        "nonmanual_input_dim":    features.shape[1],
        "nms_labels":             nms_labels,
    }


def preprocess_nonmanual():
    # 기존 json 있으면 읽기
    if os.path.exists(MANUAL_INFO_PATH):
        with open(MANUAL_INFO_PATH, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
    else:
        dataset_info = []

    # 기존 id 중복 방지용
    existing_ids = {d['id'] for d in dataset_info}

    # 전체 라벨/비디오 인덱스 구축
    label_index = build_label_index()

    # 전체 파일 대상
    targets = list(label_index.values())

    print(f"전체 전처리 대상: {len(targets)}개")

    if not targets:
        print("처리할 파일 없음")
        return

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [
            res for res in executor.map(
                process_single_file,
                targets
            ) if res
        ]

    print(f"성공: {len(results)} / {len(targets)}")

    # 기존 데이터 제거 후 새 데이터로 갱신
    result_map = {r["id"]: r for r in results}

    # 기존 dataset_info 유지하면서 업데이트
    updated_dataset = []

    # 기존 샘플 갱신
    for sample in dataset_info:
        sid = sample["id"]

        if sid in result_map:
            r = result_map[sid]

            sample["nonmanual_feature_file"] = r["nonmanual_feature_file"]
            sample["nonmanual_input_dim"] = r["nonmanual_input_dim"]
            sample["nms_labels"] = r["nms_labels"]

            updated_dataset.append(sample)

            del result_map[sid]

    # 새 샘플 추가
    for sid, r in result_map.items():
        updated_dataset.append({
            "id": sid,
            "nonmanual_feature_file": r["nonmanual_feature_file"],
            "nonmanual_input_dim": r["nonmanual_input_dim"],
            "nms_labels": r["nms_labels"],
        })

    with open(MANUAL_INFO_PATH, 'w', encoding='utf-8') as f:
        json.dump(updated_dataset, f, ensure_ascii=False, indent=4)

    if results:
        print(f"비수지 입력 차원: {results[0]['nonmanual_input_dim']}")


if __name__ == "__main__":
    preprocess_nonmanual()
