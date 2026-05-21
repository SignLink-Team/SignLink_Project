import numpy as np
import random
import os
import json

SAVE_PATH           = r"D:\새 폴더 (2)\deeprun\keypoint_data"
NONMANUAL_SAVE_PATH = os.path.join(SAVE_PATH, "nonmanual_features")

AUG_CONFIG = {
    "noise_prob":     0.5,
    "scale_prob":     0.5,
    "time_mask_prob": 0.3,
    "time_warp_prob": 0.3,
}

BASE_DIM = 180   # 60점 × 3
FULL_DIM = 540   # position + velocity + acceleration

NMS_KEYS = ["Mo1", "Mmo", "Mctr", "Ci", "Ebu", "Ebf", "Hno", "Hs"]

# 키별 증강 배수 — 활성화 비율 기준
# 5% 이하 → 5배 / 10% 이하 → 3배 / 10% 초과 → 1배
AUG_MULTIPLIER = {
    "Hs":   5,   # 1.0%
    "Ci":   5,   # 1.5%
    "Mo1":  5,   # 2.1%
    "Mctr": 3,   # 4.5%
    "Mmo":  2,   # 5.6%
    "Ebu":  1,   # 18.0%
    "Ebf":  1,   # 21.9%
    "Hno":  1,   # 21.1%
}


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------

def apply_motion_derivatives(features):
    velocity     = np.diff(features, axis=0, prepend=features[:1])
    acceleration = np.diff(velocity,  axis=0, prepend=velocity[:1])
    return np.concatenate([features, velocity, acceleration], axis=1)


def get_base(seq):
    if seq.shape[1] == BASE_DIM:
        return seq.copy()
    elif seq.shape[1] == FULL_DIM:
        return seq[:, :BASE_DIM].copy()
    else:
        raise ValueError(f"unexpected feature dim: {seq.shape[1]}")


def get_multiplier(item):
    """
    샘플에 포함된 희소 키 중 가장 높은 증강 배수 반환.
    희소 키가 없으면 1 (증강 1회).
    """
    nms_labels = item.get("nms_labels", {})
    return max(
        (AUG_MULTIPLIER[key] for key in NMS_KEYS
         if len(nms_labels.get(key, [])) > 0),
        default=1
    )


# ---------------------------------------------------------------------------
# 증강 함수
# ---------------------------------------------------------------------------

def add_noise(seq, noise_level=0.003):
    base = get_base(seq)
    base += np.random.randn(*base.shape) * noise_level
    return apply_motion_derivatives(base)


def random_scale(seq, scale_range=(0.95, 1.05)):
    base  = get_base(seq)
    scale = np.random.uniform(*scale_range)
    base *= scale
    return apply_motion_derivatives(base)


def time_mask(seq, nms_segments=None, mask_ratio=0.1):
    seq = seq.copy()
    T   = seq.shape[0]
    if T < 5:
        return seq

    if nms_segments:
        all_segs = [
            seg
            for segs in nms_segments.values()
            for seg in segs
            if seg['end_frame'] - seg['start_frame'] >= 5
        ]
        if all_segs:
            seg      = random.choice(all_segs)
            s, e     = seg['start_frame'], seg['end_frame']
            seg_len  = e - s
            mask_len = max(2, min(int(seg_len * 0.3), 5))
            mask_start = random.randint(s + 1, max(s + 1, e - mask_len))
            seq[mask_start:mask_start + mask_len] = seq[mask_start - 1]
            return seq

    mask_len = max(1, int(T * mask_ratio))
    start    = random.randint(1, max(1, T - mask_len))
    seq[start:start + mask_len] = seq[start - 1]
    return seq


def time_warp(seq, warp_range=(0.8, 1.2)):
    T       = seq.shape[0]
    factor  = np.random.uniform(*warp_range)
    new_T   = max(5, int(T * factor))
    indices = np.linspace(0, T - 1, new_T).astype(np.int32)
    return seq[indices]


def warp_nms_labels(nms_labels, original_T, warped_T):
    if original_T == 0:
        return nms_labels
    ratio  = warped_T / original_T
    warped = {}
    for key, segs in nms_labels.items():
        warped[key] = [
            {
                "start_frame": int(seg["start_frame"] * ratio),
                "end_frame":   min(int(seg["end_frame"] * ratio), warped_T - 1),
            }
            for seg in segs
            if int(seg["start_frame"] * ratio) <= min(int(seg["end_frame"] * ratio), warped_T - 1)
        ]
    return warped


def augment_nonmanual(seq, nms_labels=None):
    seq = seq.copy()

    if random.random() < AUG_CONFIG["noise_prob"]:
        seq = add_noise(seq)
    if random.random() < AUG_CONFIG["scale_prob"]:
        seq = random_scale(seq)
    if random.random() < AUG_CONFIG["time_mask_prob"]:
        seq = time_mask(seq, nms_labels)

    warped_nms = nms_labels
    if random.random() < AUG_CONFIG["time_warp_prob"]:
        original_T = seq.shape[0]
        seq        = time_warp(seq)
        if nms_labels is not None:
            warped_nms = warp_nms_labels(nms_labels, original_T, seq.shape[0])

    return seq, warped_nms


# ---------------------------------------------------------------------------
# 파일 생성
# ---------------------------------------------------------------------------

def make_aug_files():
    info_path = os.path.join(SAVE_PATH, "dataset_info.json")
    if not os.path.exists(info_path):
        print("dataset_info.json 없음 → preprocess_nms.py 먼저 실행하세요")
        return

    with open(info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)

    # 원본 샘플만 대상 (기존 aug 샘플 재증강 방지)
    targets = [
        d for d in dataset_info
        if "nonmanual_feature_file" in d and not d.get("is_aug", False)
    ]
    print(f"원본 샘플: {len(targets)}개 / 전체: {len(dataset_info)}개")

    # 키별 증강 예상량 출력
    print("\n키별 증강 예정:")
    for key, mult in AUG_MULTIPLIER.items():
        count = sum(1 for d in targets if len(d.get("nms_labels", {}).get(key, [])) > 0)
        print(f"    {key:<8} 원본 {count:>4}개 × {mult}배 = {count * mult:>5}개")
    print()

    aug_entries = []
    success = skip = 0

    for i, item in enumerate(targets):
        feature_path = os.path.join(NONMANUAL_SAVE_PATH, item['nonmanual_feature_file'])

        if not os.path.exists(feature_path):
            print(f"[{i+1}/{len(targets)}] ⚠️  파일 없음: {item['id']}")
            skip += 1
            continue

        mult       = get_multiplier(item)
        nms_labels = item.get('nms_labels', {})

        try:
            features = np.load(feature_path)

            for aug_idx in range(mult):
                aug_feat_name = item['nonmanual_feature_file'].replace(
                    '_nonmanual.npy', f'_nonmanual_aug{aug_idx}.npy'
                )
                aug_path = os.path.join(NONMANUAL_SAVE_PATH, aug_feat_name)

                # 이미 존재하면 스킵
                if os.path.exists(aug_path):
                    continue

                # 매 aug_idx 마다 다른 랜덤 증강 적용
                aug_features, aug_nms = augment_nonmanual(features, nms_labels)
                np.save(aug_path, aug_features)

                aug_entries.append({
                    **item,
                    "nonmanual_feature_file": aug_feat_name,
                    "nms_labels":             aug_nms,
                    "is_aug":                 True,
                    "aug_idx":                aug_idx,
                    "aug_source_id":          item["id"],
                })

            success += 1
            print(f"[{i+1}/{len(targets)}] ✅ {item['id']} × {mult}배 | {features.shape}")

        except Exception as e:
            print(f"[{i+1}/{len(targets)}] ❌ {item['id']} 실패: {e}")
            skip += 1

    # dataset_info.json 에 증강 샘플 추가
    if aug_entries:
        dataset_info.extend(aug_entries)
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=4)

    print(
        f"\n완료: 성공 {success}개 / 스킵 {skip}개 / 전체 {len(targets)}개\n"
        f"추가된 증강 샘플: {len(aug_entries)}개"
    )


if __name__ == "__main__":
    make_aug_files()