import cv2
import numpy as np
import os
import json
import mediapipe as mp
import concurrent.futures
from scipy.ndimage import gaussian_filter1d
from nonmanual_features import extract_nonmanual, NMS_KEYS

BASE_DIR  = r"D:\새 폴더 (2)"
TARGET_DIRS = [
    "02_NIKL_Sign Language Parallel Corpus_2023_Me1",
    "02_NIKL_Sign Language Parallel Corpus_2023_Me2",
    "02_NIKL_Sign Language Parallel Corpus_2023_Me3",
]
SAVE_PATH = "./keypoint_data"

NONMANUAL_SAVE_PATH = os.path.join(SAVE_PATH, "nonmanual_features")
os.makedirs(NONMANUAL_SAVE_PATH, exist_ok=True)

MANUAL_INFO_PATH = os.path.join(SAVE_PATH, "dataset_info.json")

def build_label_index():
    """
    TARGET_DIRS 전체를 순회해서
    {id: (label_path, video_path)} 인덱스 생성
    """
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
                    video_filename = data.get('id', '') + ".mp4"
                    # 영상은 라벨과 같은 폴더 또는 하위 폴더에 있다고 가정
                    video_path = os.path.join(root, video_filename)
                    if sample_id and os.path.exists(video_path):
                        index[sample_id] = (label_path, video_path)
                except Exception:
                    continue

    print(f"인덱스 구축 완료: {len(index)}개 파일")
    return index

def apply_motion_derivatives(features):
    velocity     = np.diff(features, axis=0, prepend=features[:1])
    acceleration = np.diff(velocity,  axis=0, prepend=velocity[:1])
    return np.concatenate([features, velocity, acceleration], axis=1)


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
        print(f"gestures 없음: {data.get('id')}")
        return None

    nms_script = data.get('nms_script', {})
    fps        = data.get('potogrf', {}).get('fps', 30)

    start_time    = min(item['start'] for item in gestures)
    end_time      = max(item['end']   for item in gestures)
    margin_frames = int(fps * 0.5)
    start_frame   = max(0, int(start_time * fps) - margin_frames)
    end_frame     = int(end_time * fps) + margin_frames

    nonmanual_seq = []
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
        prev_landmarks = None

        while cap.isOpened() and current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            frame     = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])), interpolation=cv2.INTER_LINEAR)
            results   = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                prev_landmarks = face_landmarks
            else:
                face_landmarks = prev_landmarks  # None 이면 영벡터, 아니면 직전 프레임

            nonmanual_seq.append(extract_nonmanual(face_landmarks, frame.shape))
            current_frame += 1

    if len(nonmanual_seq) < 5:
        return None

    features   = gaussian_filter1d(np.array(nonmanual_seq), sigma=1.0, axis=0)
    features   = apply_motion_derivatives(features)
    nms_labels = build_nms_labels(nms_script, len(features), fps, start_frame)

    nonmanual_file = f"{data['id']}_nonmanual.npy"
    np.save(os.path.join(NONMANUAL_SAVE_PATH, nonmanual_file), features)

    return {
        "id":                     data['id'],
        "nonmanual_feature_file": nonmanual_file,
        "nonmanual_input_dim":    features.shape[1],
        "nms_labels":             nms_labels,
    }


def preprocess_nonmanual():
    if not os.path.exists(MANUAL_INFO_PATH):
        print(f"dataset_info.json 없음: {MANUAL_INFO_PATH}")
        return

    with open(MANUAL_INFO_PATH, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)

    # dataset_info 의 id 목록
    target_ids = {d['id'] for d in dataset_info}

    # 폴더 전체 인덱스에서 target_ids 에 있는 것만 추출
    label_index = build_label_index()
    targets = [
        (sample_id, label_index[sample_id])
        for sample_id in target_ids
        if sample_id in label_index
    ]
    print(f"매칭된 파일: {len(targets)} / {len(target_ids)}개")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [
            res for res in executor.map(
                process_single_file,
                [paths for _, paths in targets]
            ) if res
        ]

    result_map = {r["id"]: r for r in results}
    for sample in dataset_info:
        if sample["id"] in result_map:
            r = result_map[sample["id"]]
            sample["nonmanual_feature_file"] = r["nonmanual_feature_file"]
            sample["nonmanual_input_dim"]    = r["nonmanual_input_dim"]
            sample["nms_labels"]             = r["nms_labels"]

    with open(MANUAL_INFO_PATH, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=4)

    print(
        f"비수지 전처리 완료.\n"
        f"성공: {len(results)} / {len(targets)}\n")
    if not results:
        print("성공한 파일이 없습니다. 경로/파일 확인 필요")
        print(BASE_DIR)
        print(os.path.exists(BASE_DIR))
        total_mp4 = 0
        for root, _, files in os.walk(BASE_DIR):
            for fname in files:
                if fname.endswith(".mp4"):
                    total_mp4 += 1

        print("전체 mp4 개수:", total_mp4)
        return
    print(f"비수지 입력 차원: {results[0]['nonmanual_input_dim']}"
    )
    
    


if __name__ == "__main__":
    preprocess_nonmanual()