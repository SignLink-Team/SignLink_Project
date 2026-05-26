import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 방금 분리한 파일들에서 임포트
from model import SignLanguageModel
from dataset import stratified_sentence_split, SignLanguageInMemoryDataset, ctc_collate_fn

def train_model():
    # ==========================
    # 1. 설정 및 경로 지정
    # ==========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    DATA_DIR = "./keypoint_data"
    INFO_FILE = os.path.join(DATA_DIR, "dataset_info.json")
    FEATURE_DIR = DATA_DIR 
    GLOSS_DICT_FILE = os.path.join(DATA_DIR, "gloss_dict.json")
    MAPPING_FILE = os.path.join(DATA_DIR, "gloss_mapping.json")

    # ==========================
    # 2. 메타데이터 및 사전 로드 (UNK 안전장치 추가)
    # ==========================
    with open(INFO_FILE, 'r', encoding='utf-8') as f:
        all_info = json.load(f)
        
    with open(GLOSS_DICT_FILE, 'r', encoding='utf-8') as f:
        gloss_to_idx = json.load(f)
        
    # 🌟 [수정됨] 사전에 <UNK>가 없다면 정식 인덱스로 추가 (마이너스 Loss 원인 해결)
    if "<UNK>" not in gloss_to_idx:
        gloss_to_idx["<UNK>"] = len(gloss_to_idx)
        print(f"⚠️ 사전에 <UNK>가 없어 정식 인덱스({gloss_to_idx['<UNK>']})로 추가했습니다.")
        
    mapping_dict = {}
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            mapping_dict = json.load(f)

    # ==========================
    # 3. 층화 분할 (Train/Val/Test)
    # ==========================
    print(">>> 층화 추출(Stratified Split) 진행 중...")
    train_info, val_info, test_info = stratified_sentence_split(
        all_info, mapping_dict, train_ratio=0.8, val_ratio=0.1
    )

    # ==========================
    # 4. In-memory Dataset 로드
    # ==========================
    train_dataset = SignLanguageInMemoryDataset(train_info, FEATURE_DIR, gloss_to_idx, mapping_dict)
    val_dataset = SignLanguageInMemoryDataset(val_info, FEATURE_DIR, gloss_to_idx, mapping_dict)
    test_dataset = SignLanguageInMemoryDataset(test_info, FEATURE_DIR, gloss_to_idx, mapping_dict)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=ctc_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=ctc_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=ctc_collate_fn)

    # ==========================
    # 5. 모델 초기화
    # ==========================
    vocab_size = len(gloss_to_idx)
    print(f"최종 어휘 사전(Vocabulary) 크기: {vocab_size}")
    
    model = SignLanguageModel(input_dim=783, hidden_dim=512, num_classes=vocab_size).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # ==========================
    # 6. 학습 루프 (조기 종료 추가)
    # ==========================
    EPOCHS = 150  # 에포크를 넉넉하게 잡음
    best_val_loss = float('inf')
    
    patience_limit = 10  # 10번 연속 갱신 안되면 조기 종료
    patience_check = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for features, targets, feat_lengths, tgt_lengths in pbar:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(features).log_softmax(2).permute(1, 0, 2)
            
            loss = criterion(logits, targets, feat_lengths, tgt_lengths)
            if torch.isnan(loss) or torch.isinf(loss): continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets, feat_lengths, tgt_lengths in val_loader:
                features, targets = features.to(device), targets.to(device)
                logits = model(features).log_softmax(2).permute(1, 0, 2)
                loss = criterion(logits, targets, feat_lengths, tgt_lengths)
                val_loss += loss.item()
                
        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))
        scheduler.step(avg_val_loss)
        
        print(f"Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 🌟 [수정됨] 조기 종료 로직 적용
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_sign_model.pth')
            print("🌟 Best Model Saved!")
            patience_check = 0  # 갱신되었으므로 카운트 초기화
        else:
            patience_check += 1
            print(f"⚠️ Early Stopping 카운트: {patience_check} / {patience_limit}")
            
            if patience_check >= patience_limit:
                print(f"\n🛑 조기 종료 발동! 더 이상 성능이 개선되지 않아 {epoch+1} 에포크에서 학습을 강제 중단합니다.")
                break

    # ==========================
    # 7. Test 평가 (최고 성능 모델 로드)
    # ==========================
    print("\n>>> 학습 종료! Test Set으로 최종 성능 검증 시작...")
    model.load_state_dict(torch.load('best_sign_model.pth'))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for features, targets, feat_lengths, tgt_lengths in tqdm(test_loader, desc="[Test]"):
            features, targets = features.to(device), targets.to(device)
            logits = model(features).log_softmax(2).permute(1, 0, 2)
            loss = criterion(logits, targets, feat_lengths, tgt_lengths)
            test_loss += loss.item()
            
    print(f"✅ 최종 Test Loss: {test_loss / max(1, len(test_loader)):.4f}")

if __name__ == "__main__":
    train_model()