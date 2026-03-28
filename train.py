import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import os
from torch.nn.utils.rnn import pad_sequence
from model import SignLanguageModel

# --- 상대 경로 설정 ---
SAVE_PATH = "./keypoint_data"

class SignDataset(Dataset):
    def __init__(self, data_info_path, dict_path, features_dir):
        with open(data_info_path, 'r', encoding='utf-8') as f:
            self.data_info = json.load(f)
        with open(dict_path, 'r', encoding='utf-8') as f:
            self.gloss_dict = json.load(f)
        self.features_dir = features_dir

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        item = self.data_info[idx]
        feature_path = os.path.join(self.features_dir, item['feature_file'])
        features = np.load(feature_path)
        
        target_indices = [self.gloss_dict[gloss] for gloss in item['gloss_sequence'] if gloss in self.gloss_dict]
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target_indices, dtype=torch.long)

def collate_fn(batch):
    features, targets = zip(*batch)
    feature_lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)
    target_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)
    
    padded_features = pad_sequence(features, batch_first=True)
    padded_targets = pad_sequence(targets, batch_first=True)
    
    return padded_features, padded_targets, feature_lengths, target_lengths

def train_model():
    dataset_info_file = os.path.join(SAVE_PATH, "dataset_info.json")
    dict_file = os.path.join(SAVE_PATH, "gloss_dict.json")
    
    if not os.path.exists(dataset_info_file) or not os.path.exists(dict_file):
        print(f"오류: {SAVE_PATH} 에 전처리된 데이터가 없습니다. preprocess.py를 먼저 실행하세요.")
        return

    full_dataset = SignDataset(
        data_info_path=dataset_info_file,
        dict_path=dict_file,
        features_dir=SAVE_PATH
    )
    
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    with open(dict_file, 'r', encoding='utf-8') as f:
        gloss_dict = json.load(f)
    num_classes = len(gloss_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습을 시작합니다. 사용 장치: {device}")
    
    model = SignLanguageModel(input_dim=255, hidden_dim=256, num_classes=num_classes).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    best_val_loss = float('inf')
    model_save_path = os.path.join(SAVE_PATH, "sign_model_best.pth")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, targets, feat_lengths, tgt_lengths in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(features)
            logits = logits.log_softmax(2).permute(1, 0, 2)
            
            loss = criterion(logits, targets, feat_lengths, tgt_lengths)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets, feat_lengths, tgt_lengths in val_loader:
                features, targets = features.to(device), targets.to(device)
                logits = model(features)
                logits = logits.log_softmax(2).permute(1, 0, 2)
                loss = criterion(logits, targets, feat_lengths, tgt_lengths)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / max(1, len(val_loader))
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Best model saved at {model_save_path}")

    print("학습 완료!")

if __name__ == "__main__":
    train_model()