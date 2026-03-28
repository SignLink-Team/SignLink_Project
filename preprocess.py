import torch
import numpy as np
import json
import os
import random
import difflib
from model import SignLanguageModel

# [추가됨] 방향성 동사 라벨(예: 돕다1_1_2)을 사람이 읽기 편하게 변환하는 함수
def format_gloss(gloss):
    parts = gloss.split('_')
    if len(parts) == 3:
        base, src, tgt = parts
        return f"{base} ({src}→{tgt})"
    return gloss

def decode_predictions(predictions, idx_to_gloss):
    decoded_sequence = []
    previous_idx = -1
    for idx in predictions:
        if idx != previous_idx and idx != 0:  # 0은 <blank> 토큰
            decoded_sequence.append(idx_to_gloss[idx])
        previous_idx = idx
    return decoded_sequence

def load_model(model_path, dict_path, device):
    with open(dict_path, 'r', encoding='utf-8') as f:
        gloss_dict = json.load(f)
        
    idx_to_gloss = {v: k for k, v in gloss_dict.items()}
    num_classes = len(gloss_dict)

    # [수정됨] 모델 구조 변경점 반영 (num_layers=3, dropout=0.5)
    model = SignLanguageModel(
        input_dim=255, 
        hidden_dim=256, 
        num_layers=3,         # 변경된 층 수 반영
        num_classes=num_classes, 
        dropout=0.5           # 변경된 드롭아웃 반영
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, idx_to_gloss, gloss_dict

def predict_single_file(feature_numpy_array, model, device, idx_to_gloss):
    features_tensor = torch.tensor(feature_numpy_array, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(features_tensor)
        probabilities = logits.softmax(2)
        predictions = torch.argmax(probabilities, dim=2)
        
    predicted_indices = predictions[0].cpu().numpy()
    return decode_predictions(predicted_indices, idx_to_gloss)

def analyze_prediction(true_seq, pred_seq, gloss_dict):
    """정답과 예측 결과를 비교하여 상세한 분석을 출력합니다."""
    print("\n[상세 분석 결과]")
    matcher = difflib.SequenceMatcher(None, true_seq, pred_seq)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for w in true_seq[i1:i2]:
                print(f"  [O] '{format_gloss(w)}' : 정확히 일치함!")
                
        elif tag == 'delete':
            for w in true_seq[i1:i2]:
                status = "학습 사전에 존재함" if w in gloss_dict else "사전에 없는 미학습 단어"
                print(f"  [X] '{format_gloss(w)}' : 모델이 인식하지 못함 (원인: {status})")
                
        elif tag == 'insert':
            for w in pred_seq[j1:j2]:
                print(f"  [!] '{format_gloss(w)}' : 모델이 잘못 예측하여 끼워넣음 (오인식)")
                
        elif tag == 'replace':
            limit = min(i2 - i1, j2 - j1)
            for k in range(limit):
                w1, w2 = true_seq[i1+k], pred_seq[j1+k]
                status = "학습 사전에 존재함" if w1 in gloss_dict else "미학습 단어"
                print(f"  [X] 정답 '{format_gloss(w1)}' ({status}) -> 예측 '{format_gloss(w2)}' (오답으로 교체됨)")
            
            if (i2 - i1) > (j2 - j1):
                for w in true_seq[i1+limit:i2]:
                    status = "학습 사전에 존재함" if w in gloss_dict else "미학습 단어"
                    print(f"  [X] '{format_gloss(w)}' : 모델이 인식하지 못함 (원인: {status})")
            elif (i2 - i1) < (j2 - j1):
                for w in pred_seq[j1+limit:j2]:
                    print(f"  [!] '{format_gloss(w)}' : 모델이 잘못 예측하여 끼워넣음 (오인식)")

if __name__ == "__main__":
    # --- 상대 경로 설정 ---
    BASE_DIR = "./keypoint_data"
    
    dataset_info_path = os.path.join(BASE_DIR, "dataset_info.json")
    best_model_path = os.path.join(BASE_DIR, "sign_model_best.pth")
    dict_file_path = os.path.join(BASE_DIR, "gloss_dict.json")
    
    if not os.path.exists(dataset_info_path):
        print(f"오류: 데이터 정보 파일이 없습니다. ({dataset_info_path})")
    elif not os.path.exists(best_model_path):
        print(f"오류: 학습된 모델 파일이 없습니다. ({best_model_path})")
    elif not os.path.exists(dict_file_path):
        print(f"오류: 단어 사전 파일이 없습니다. ({dict_file_path})")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, idx_to_gloss, gloss_dict = load_model(best_model_path, dict_file_path, device)
        
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
            
        # 테스트할 샘플 개수
        num_test_samples = 5
        test_samples = random.sample(dataset_info, min(num_test_samples, len(dataset_info)))
        correct_count = 0
        
        print(f"총 {num_test_samples}개의 샘플에 대해 무작위 예측 테스트를 시작합니다...\n")
        print("="*50)
        
        for i, sample in enumerate(test_samples, 1):
            feature_file = sample['feature_file']
            true_sequence = sample['gloss_sequence']
            
            feature_path = os.path.join(BASE_DIR, feature_file)
            
            if not os.path.exists(feature_path):
                print(f"[{i}] 파일 누락: {feature_file}")
                continue
                
            sample_features = np.load(feature_path)
            predicted_sequence = predict_single_file(sample_features, model, device, idx_to_gloss)
            
            # 출력할 때 읽기 편하도록 format_gloss 적용
            true_str = " ".join([format_gloss(w) for w in true_sequence])
            pred_str = " ".join([format_gloss(w) for w in predicted_sequence])
            
            print(f"--- 테스트 샘플 {i} ({sample['id']}) ---")
            print(f"▶ 실제 정답: {true_str}")
            print(f"▶ 모델 예측: {pred_str}")
            
            if true_sequence == predicted_sequence:
                print("\n결과: 완벽하게 일치합니다! [O]\n")
                correct_count += 1
            else:
                analyze_prediction(true_sequence, predicted_sequence, gloss_dict)
                print()
            print("-" * 50)
                
        print(f"완벽히 일치한 문장: {correct_count} / {len(test_samples)}")