import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import numpy as np
import json
import os
import random
from torch.nn.utils.rnn import pad_sequence
from model import SignLanguageModel

# 상수 설정
SAVE_PATH = "./keypoint_data"
INPUT_DIM = 711 
FILE_KEYS = ['feature_file', 'gray_file', 'aug_file', 'gray_aug_file']
VAL_RATIO = 0.2
MIN_TRAIN_SAMPLES = 2 # 각 단어당 학습셋 최소 포함 개수 (확장된 샘플 기준)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SignDataset(Dataset):
    def __init__(self, samples, gloss_dict, features_dir):
        self.samples = samples
        self.gloss_dict = gloss_dict
        self.features_dir = features_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        feature_path = os.path.join(self.features_dir, item['feature_file'])
        features = np.load(feature_path)
        target_indices = [self.gloss_dict[g] for g in item['gloss_sequence'] if g in self.gloss_dict]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target_indices, dtype=torch.long)

def collate_fn(batch):
    features, targets = zip(*batch)
    feature_lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)
    target_lengths  = torch.tensor([t.size(0) for t in targets],  dtype=torch.long)
    padded_features = pad_sequence(features, batch_first=True)
    padded_targets  = pad_sequence(targets,  batch_first=True)
    return padded_features, padded_targets, feature_lengths, target_lengths

def get_sample_weights(samples):
    freq = {}
    for s in samples:
        for g in s['gloss_sequence']:
            freq[g] = freq.get(g, 0) + 1
    weights = []
    for s in samples:
        if not s['gloss_sequence']:
            weights.append(0.01)
            continue
        min_f = min([freq.get(g, 1) for g in s['gloss_sequence']])
        weights.append(1.0 / np.sqrt(min_f))
    return weights

def train_model():
    set_seed()
    dataset_info_file = os.path.join(SAVE_PATH, "dataset_info.json")
    dict_file         = os.path.join(SAVE_PATH, "gloss_dict.json")
    
    if not os.path.exists(dataset_info_file) or not os.path.exists(dict_file):
        print(f"오류: {SAVE_PATH}에 데이터가 없습니다.")
        return

    with open(dataset_info_file, 'r', encoding='utf-8') as f:
        raw_info = json.load(f)
    with open(dict_file, 'r', encoding='utf-8') as f:
        gloss_dict = json.load(f)

    # --- [수정] 1. 먼저 4종 데이터로 확장(펼치기) ---
    all_expanded_samples = []
    for item in raw_info:
        for key in FILE_KEYS:
            if key in item:
                path = os.path.join(SAVE_PATH, item[key])
                if os.path.exists(path):
                    all_expanded_samples.append({
                        "feature_file": item[key],
                        "gloss_sequence": item["gloss_sequence"]
                    })

    # --- [수정] 2. 확장된 샘플을 기준으로 단어별 인덱스 수집 ---
    gloss_to_sample_indices = {}
    for idx, sample in enumerate(all_expanded_samples):
        for g in sample['gloss_sequence']:
            if g not in gloss_to_sample_indices:
                gloss_to_sample_indices[g] = []
            gloss_to_sample_indices[g].append(idx)

    # --- [수정] 3. 모든 단어가 최소 2개씩 Train에 포함되도록 분할 ---
    train_indices = set()
    
    # 각 단어별로 최소 2개 샘플 선점
    for gloss, indices in gloss_to_sample_indices.items():
        random.shuffle(indices)
        take_count = min(len(indices), MIN_TRAIN_SAMPLES)
        for i in range(take_count):
            train_indices.add(indices[i])

    # 나머지 샘플 8:2 분할
    remaining_indices = [i for i in range(len(all_expanded_samples)) if i not in train_indices]
    random.shuffle(remaining_indices)
    
    target_train_total = int(len(all_expanded_samples) * (1 - VAL_RATIO))
    num_to_add = max(0, target_train_total - len(train_indices))
    
    final_train_indices = list(train_indices) + remaining_indices[:num_to_add]
    final_val_indices = remaining_indices[num_to_add:]

    # [추가] 검증셋으로 분류된 파일명 리스트 생성
    val_file_names = [all_expanded_samples[i]['feature_file'] for i in final_val_indices]
    
    # [추가] 검증셋 파일 리스트를 JSON으로 저장
    val_indices_path = os.path.join(SAVE_PATH, "val_indices.json")
    with open(val_indices_path, 'w', encoding='utf-8') as f:
        json.dump(val_file_names, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 검증셋 파일 리스트 저장 완료: {val_indices_path} ({len(val_file_names)}개)")

    train_samples = [all_expanded_samples[i] for i in final_train_indices]
    val_samples = [all_expanded_samples[i] for i in final_val_indices]

    print(f"[확장 후 분할 결과] 총 확장 샘플: {len(all_expanded_samples)}")
    print(f"Train 샘플: {len(train_samples)} / Val 샘플: {len(val_samples)}")

    # Dataset & Loader
    train_dataset = SignDataset(train_samples, gloss_dict, SAVE_PATH)              
    val_dataset = SignDataset(val_samples, gloss_dict, SAVE_PATH)

    train_weights = get_sample_weights(train_samples)
    sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # [이후 학습 로직은 기존과 동일하게 유지]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(gloss_dict)
    model = SignLanguageModel(input_dim=INPUT_DIM, hidden_dim=256, num_layers=3, num_classes=num_classes, dropout=0.5).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    epochs = 70
    best_val_loss = float('inf')
    model_save_path = os.path.join(SAVE_PATH, "sign_model_best.pth")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, targets, feat_lengths, tgt_lengths in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(features).log_softmax(2).permute(1, 0, 2)
            loss = criterion(logits, targets, feat_lengths, tgt_lengths)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets, feat_lengths, tgt_lengths in val_loader:
                features, targets = features.to(device), targets.to(device)
                logits = model(features).log_softmax(2).permute(1, 0, 2)
                loss = criterion(logits, targets, feat_lengths, tgt_lengths)
                val_loss += loss.item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / max(1, len(val_loader))
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Best model saved!")

    print("학습 완료!")

if __name__ == "__main__":
    train_model()
