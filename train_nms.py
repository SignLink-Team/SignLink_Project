"""
train_nms.py
------------
비수지(Non-manual) 분류기 학습 코드

구조:
    Dataset     → nms_labels 구간 → dense (T, 8) 변환
    DataLoader  → WeightedRandomSampler (희소 클래스 보정)
    Model       → Transformer Encoder + Linear(8) head
    Loss        → FocalBCELoss (pos_weight + focal)
    평가        → 키별 F1 score
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score
from tqdm import tqdm
import math

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

SAVE_PATH           = r"D:\새 폴더 (2)\deeprun\keypoint_data"
NONMANUAL_SAVE_PATH = os.path.join(SAVE_PATH, "nonmanual_features")
INFO_PATH           = os.path.join(SAVE_PATH, "dataset_info.json")
CKPT_DIR            = os.path.join(SAVE_PATH, "checkpoints_nms")
os.makedirs(CKPT_DIR, exist_ok=True)

NMS_KEYS   = ["Mo1", "Mmo", "Mctr", "Ci", "Ebu", "Ebf", "Hno", "Hs"]
NUM_LABELS = len(NMS_KEYS)

# 전처리 결과 기반 pos_weight (비활성 / 활성 비율)
TOTAL_FRAMES = 884_658
POS_COUNTS   = {
    "Mo1":  18_876,
    "Mmo":  49_551,
    "Mctr": 39_691,
    "Ci":   13_671,
    "Ebu": 159_558,
    "Ebf": 193_621,
    "Hno": 186_705,
    "Hs":    8_509,
}
POS_WEIGHT = torch.tensor(
    [math.sqrt(TOTAL_FRAMES / POS_COUNTS[k]) for k in NMS_KEYS],
    dtype=torch.float32
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = {
    "input_dim":   540,      # 비수지 피처 차원 (180 × 3)
    "d_model":     128,
    "nhead":         4,
    "num_layers":    3,
    "dropout":     0.2,
    "lr":          1e-4,
    "weight_decay":1e-4,
    "epochs":       60,
    "batch_size":    16,
    "val_ratio":    0.15,
    "focal_gamma":  2.0,
    "rare_keys":   ["Ci", "Hs", "Mo1", "Mctr"],  # 희소 클래스 (샘플 가중치 상향)
    "rare_weight":  4.0,
    "seed":         42,
}


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def segments_to_dense(nms_labels, total_frames):
    """구간 구조 → (T, 8) dense label"""
    label = np.zeros((total_frames, NUM_LABELS), dtype=np.float32)
    for i, key in enumerate(NMS_KEYS):
        for seg in nms_labels.get(key, []):
            s = max(0, seg["start_frame"])
            e = min(total_frames - 1, seg["end_frame"])
            label[s:e + 1, i] = 1.0
    return label


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NMSDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        feat_path = os.path.join(NONMANUAL_SAVE_PATH, item["nonmanual_feature_file"])
        features  = np.load(feat_path).astype(np.float32)   # (T, 540)
        T         = len(features)
        label     = segments_to_dense(item["nms_labels"], T) # (T, 8)

        return (
            torch.from_numpy(features),   # (T, 540)
            torch.from_numpy(label),      # (T, 8)
        )


def collate_fn(batch):
    """가변 길이 시퀀스 패딩"""
    feats, labels = zip(*batch)
    lengths       = torch.tensor([f.shape[0] for f in feats])
    feats_pad     = pad_sequence(feats,  batch_first=True, padding_value=0.0)  # (B, T_max, 540)
    labels_pad    = pad_sequence(labels, batch_first=True, padding_value=-1.0) # (B, T_max, 8) — -1은 패딩 마스크
    return feats_pad, labels_pad, lengths


# ---------------------------------------------------------------------------
# 샘플 가중치 (WeightedRandomSampler)
# ---------------------------------------------------------------------------

def get_sample_weights(samples):
    """희소 클래스 포함 샘플에 higher weight"""
    weights = []
    for item in samples:
        has_rare = any(
            len(item["nms_labels"].get(k, [])) > 0
            for k in CFG["rare_keys"]
        )
        weights.append(CFG["rare_weight"] if has_rare else 1.0)
    return weights


# ---------------------------------------------------------------------------
# Loss: Focal BCE
# ---------------------------------------------------------------------------

class FocalBCELoss(nn.Module):
    def __init__(self, pos_weight, gamma=2.0):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight  # (num_labels,)

    def forward(self, pred, target, mask):
        """
        pred   : (B, T, 8)
        target : (B, T, 8)  — 패딩은 -1
        mask   : (B, T)     — 유효 프레임 True
        """
        pw = self.pos_weight.to(pred.device)

        # 패딩 제외
        pred_valid   = pred[mask]    # (N, 8)
        target_valid = target[mask]  # (N, 8)

        bce  = F.binary_cross_entropy_with_logits(
            pred_valid, target_valid,
            pos_weight=pw,
            reduction="none"
        )
        prob = torch.sigmoid(pred_valid)
        p_t  = target_valid * prob + (1 - target_valid) * (1 - prob)
        loss = ((1 - p_t) ** self.gamma * bce).mean()
        return loss


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NMSClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_labels, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head       = nn.Linear(d_model, num_labels)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        """
        x    : (B, T, input_dim)
        mask : (B, T) — 패딩 위치 True
        반환 : (B, T, num_labels) logits
        """
        x = self.dropout(self.input_proj(x))
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.head(x)


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for feats, labels, lengths in loader:
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)

        T_max   = feats.shape[1]
        pad_mask = torch.arange(T_max, device=DEVICE).unsqueeze(0) >= lengths.unsqueeze(1).to(DEVICE)
        valid_mask = ~pad_mask  # (B, T)

        logits = model(feats, src_key_padding_mask=pad_mask)
        loss   = criterion(logits, labels.clamp(min=0), valid_mask)
        total_loss += loss.item()

        preds   = (torch.sigmoid(logits) > 0.5).float()
        all_preds.append(preds[valid_mask].cpu().numpy())
        all_targets.append(labels[valid_mask].cpu().numpy())

    all_preds   = np.concatenate(all_preds,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 키별 F1
    f1_per_key = {}
    for i, key in enumerate(NMS_KEYS):
        f1_per_key[key] = f1_score(
            all_targets[:, i], all_preds[:, i],
            zero_division=0
        )
    macro_f1 = np.mean(list(f1_per_key.values()))

    return total_loss / len(loader), f1_per_key, macro_f1


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------

def train():
    set_seed(CFG["seed"])

    # 데이터 로드
    with open(INFO_PATH, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)

    # 비수지 전처리 완료 + aug 샘플 포함
    all_samples = [
        d for d in dataset_info
        if "nonmanual_feature_file" in d and "nms_labels" in d
    ]
    print(f"전체 샘플: {len(all_samples)}개")

    # train / val 분할 — aug 샘플은 train 에만
    orig    = [d for d in all_samples if not d.get("is_aug", False)]
    aug     = [d for d in all_samples if d.get("is_aug", False)]

    random.shuffle(orig)
    n_val    = int(len(orig) * CFG["val_ratio"])
    val_data = orig[:n_val]
    trn_data = orig[n_val:] + aug   # aug 는 train 에만

    print(f"train: {len(trn_data)}개 (aug {len(aug)}개 포함) / val: {len(val_data)}개")

    trn_set = NMSDataset(trn_data)
    val_set = NMSDataset(val_data)

    # WeightedRandomSampler
    sample_weights = get_sample_weights(trn_data)
    sampler        = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    trn_loader = DataLoader(
        trn_set, batch_size=CFG["batch_size"],
        sampler=sampler, collate_fn=collate_fn,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=CFG["batch_size"],
        shuffle=False, collate_fn=collate_fn,
        num_workers=4, pin_memory=True,
    )

    # 모델
    model = NMSClassifier(
        input_dim  = CFG["input_dim"],
        d_model    = CFG["d_model"],
        nhead      = CFG["nhead"],
        num_layers = CFG["num_layers"],
        num_labels = NUM_LABELS,
        dropout    = CFG["dropout"],
    ).to(DEVICE)

    criterion = FocalBCELoss(POS_WEIGHT, gamma=CFG["focal_gamma"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"]
    )

    best_f1   = 0.0
    best_ckpt = os.path.join(CKPT_DIR, "best_nms.pt")

    for epoch in range(1, CFG["epochs"] + 1):
        # --- train ---
        model.train()
        trn_loss = 0.0

        for feats, labels, lengths in tqdm(trn_loader, desc=f"Epoch {epoch:02d} train", leave=False):
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)

            T_max    = feats.shape[1]
            pad_mask = torch.arange(T_max, device=DEVICE).unsqueeze(0) >= lengths.unsqueeze(1).to(DEVICE)
            valid_mask = ~pad_mask

            optimizer.zero_grad()
            logits = model(feats, src_key_padding_mask=pad_mask)
            loss   = criterion(logits, labels.clamp(min=0), valid_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            trn_loss += loss.item()

        scheduler.step()
        trn_loss /= len(trn_loader)

        # --- val ---
        val_loss, f1_per_key, macro_f1 = evaluate(model, val_loader, criterion)

        # 체크포인트
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save({
                "epoch":    epoch,
                "model":    model.state_dict(),
                "cfg":      CFG,
                "nms_keys": NMS_KEYS,
                "macro_f1": macro_f1,
            }, best_ckpt)

        print(
            f"[{epoch:02d}/{CFG['epochs']}] "
            f"trn_loss: {trn_loss:.4f}  val_loss: {val_loss:.4f}  "
            f"macro_F1: {macro_f1:.4f}  best: {best_f1:.4f}"
        )
        print("  " + "  ".join(f"{k}:{v:.3f}" for k, v in f1_per_key.items()))

    print(f"\n학습 완료. best macro F1: {best_f1:.4f} → {best_ckpt}")


if __name__ == "__main__":
    train()