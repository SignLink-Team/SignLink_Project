# predict.py 수정본
import torch
import numpy as np
import json
import os
import random
import difflib
from model import SignLanguageModel

INPUT_DIM = 255
HIDDEN_DIM = 256
NUM_LAYERS = 2 
DROPOUT = 0.1

def format_gloss(gloss):
    parts = gloss.split('_')
    if len(parts) == 3:
        return f"{parts[0]} ({parts[1]}->{parts[2]})"
    return gloss

def decode_predictions(predictions, idx_to_gloss):
    decoded_sequence = []
    previous_idx = -1
    for idx in predictions:
        if idx != previous_idx and idx != 0:
            decoded_sequence.append(idx_to_gloss[idx])
        previous_idx = idx
    return decoded_sequence

def load_model(model_path, dict_path, device):
    with open(dict_path, 'r', encoding='utf-8') as f:
        gloss_dict = json.load(f)
    idx_to_gloss = {v: k for k, v in gloss_dict.items()}
    model = SignLanguageModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, len(gloss_dict), DROPOUT).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, idx_to_gloss, gloss_dict

def analyze_detailed_prediction(true_seq, pred_seq, gloss_dict):
    print("\n[분석 결과]")
    matcher = difflib.SequenceMatcher(None, true_seq, pred_seq)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for w in true_seq[i1:i2]:
                print(f"  일치: {format_gloss(w)}")
                
        elif tag == 'delete':
            for w in true_seq[i1:i2]:
                if w not in gloss_dict:
                    print(f"  미학습 단어: {format_gloss(w)}")
                else:
                    print(f"  모델 인식 못함(누락): {format_gloss(w)}")
                    
        elif tag == 'replace':
            for k in range(min(i2-i1, j2-j1)):
                t_word = true_seq[i1+k]
                p_word = pred_seq[j1+k]
                print(f"  학습했지만 틀림(오답): {format_gloss(t_word)} -> 예측: {format_gloss(p_word)}")
            
            if (i2-i1) > (j2-j1):
                for w in true_seq[i1+(j2-j1):i2]:
                    print(f"  모델 인식 못함(누락): {format_gloss(w)}")
            elif (i2-i1) < (j2-j1):
                for w in pred_seq[j1+(i2-i1):j2]:
                    print(f"  잘못된 추가 예측: {format_gloss(w)}")
                    
        elif tag == 'insert':
            for w in pred_seq[j1:j2]:
                print(f"  잘못된 추가 예측: {format_gloss(w)}")

if __name__ == "__main__":
    BASE_DIR = "./keypoint_data"
    # [수정] dataset_info.json 대신 train.py에서 분리해둔 val_dataset_info.json을 불러옵니다!
    val_info_path = os.path.join(BASE_DIR, "val_dataset_info.json")
    best_model_path = os.path.join(BASE_DIR, "sign_model_best.pth")
    dict_file_path = os.path.join(BASE_DIR, "gloss_dict.json")
    
    if os.path.exists(val_info_path) and os.path.exists(best_model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, idx_to_gloss, gloss_dict = load_model(best_model_path, dict_file_path, device)
        
        # 순수한 테스트 셋 리스트 불러오기
        with open(val_info_path, 'r', encoding='utf-8') as f:
            test_set = json.load(f)
            
        # 순수한 테스트 셋 안에서 무작위로 5개 추출하여 예측
        test_samples = random.sample(test_set, min(5, len(test_set)))
        
        for i, sample in enumerate(test_samples, 1):
            feature_path = os.path.join(BASE_DIR, sample['feature_file'])
            true_sequence = sample['gloss_sequence']
            
            features = np.load(feature_path)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(features_tensor)
                pred_indices = torch.argmax(logits.softmax(2), dim=2)[0].cpu().numpy()
            predicted_sequence = decode_predictions(pred_indices, idx_to_gloss)
            
            print(f"\n샘플 {i} ID: {sample['id']}")
            print(f"정답: {' '.join([format_gloss(w) for w in true_sequence])}")
            print(f"예측: {' '.join([format_gloss(w) for w in predicted_sequence])}")
            
            analyze_detailed_prediction(true_sequence, predicted_sequence, gloss_dict)
            print("-" * 50)
    else:
        print(f"필요한 파일이 없습니다. {val_info_path} 또는 {best_model_path}를 확인하세요. train.py를 다시 실행해야 합니다.")