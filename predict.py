import os
import json
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm

# 기존 모듈 임포트
from model import SignLanguageModel
from dataset import stratified_sentence_split

# ==========================================
# 1. 평가 지표 계산 함수들
# ==========================================
def calculate_wer(ref, hyp):
    """
    편집 거리(Levenshtein Distance) 알고리즘을 이용한 WER 계산
    ref: 정답 단어 리스트, hyp: 예측 단어 리스트
    """
    # 초기화
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    for i in range(len(ref) + 1): d[i][0] = i
    for j in range(len(hyp) + 1): d[0][j] = j

    # DP 계산
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # 삭제 (Deletion)
                d[i][j - 1] + 1,      # 삽입 (Insertion)
                d[i - 1][j - 1] + cost # 대체 (Substitution)
            )
            
    # WER = 편집 거리 / 정답 단어 개수
    distance = d[len(ref)][len(hyp)]
    return distance, len(ref)

def calculate_token_f1(ref, hyp):
    """
    순서에 상관없이 단어(토큰)를 얼마나 잘 캐치했는지 측정하는 F1-Score (Bag-of-Words 기반)
    """
    ref_counts = Counter(ref)
    hyp_counts = Counter(hyp)
    
    # 교집합(맞춘 단어) 개수 계산
    overlap = sum(min(ref_counts[w], hyp_counts[w]) for w in ref_counts)
    
    precision = overlap / len(hyp) if len(hyp) > 0 else 0.0
    recall = overlap / len(ref) if len(ref) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
        
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

def ctc_greedy_decode(predictions, idx_to_gloss, blank_idx=0):
    """모델의 출력을 텍스트로 디코딩 (중복 및 blank 제거)"""
    decoded = []
    prev_idx = -1
    for idx in predictions:
        if idx != prev_idx and idx != blank_idx:
            # 사전에 없는 경우 방어 (혹시 모를 에러 대비)
            decoded.append(idx_to_gloss.get(idx, "<UNK>"))
        prev_idx = idx
    return decoded

# ==========================================
# 2. 메인 평가 루프
# ==========================================
def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 경로 설정
    DATA_DIR = "./keypoint_data"
    INFO_FILE = os.path.join(DATA_DIR, "dataset_info.json")
    FEATURE_DIR = DATA_DIR 
    GLOSS_DICT_FILE = os.path.join(DATA_DIR, "gloss_dict.json")
    MAPPING_FILE = os.path.join(DATA_DIR, "gloss_mapping.json")
    MODEL_PATH = "best_sign_model.pth"

    # 1. 메타데이터 및 사전 로드
    with open(INFO_FILE, 'r', encoding='utf-8') as f:
        all_info = json.load(f)
        
    with open(GLOSS_DICT_FILE, 'r', encoding='utf-8') as f:
        gloss_to_idx = json.load(f)
        
    if "<UNK>" not in gloss_to_idx:
        gloss_to_idx["<UNK>"] = len(gloss_to_idx)
        
    mapping_dict = {}
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            mapping_dict = json.load(f)

    # 2. Test Set 분리 (학습 때와 완벽히 동일한 환경 구성)
    print(">>> 층화 추출을 통한 Test Set 분리 중...")
    _, _, test_info = stratified_sentence_split(all_info, mapping_dict, train_ratio=0.8, val_ratio=0.1)

    # 3. 모델 로드
    idx_to_gloss = {v: k for k, v in gloss_to_idx.items()}
    vocab_size = len(gloss_to_idx)
    
    model = SignLanguageModel(input_dim=783, hidden_dim=512, num_classes=vocab_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 4. 성능 척도 초기화
    total_samples = len(test_info)
    exact_match_count = 0
    total_edit_distance = 0
    total_ref_words = 0
    
    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0

    print(f"\n🚀 총 {total_samples}개의 Test Set 평가 시작...\n")

    # 5. 예측 및 평가 진행
    for i, info in enumerate(tqdm(test_info, desc="Evaluating")):
        feat_path = os.path.join(FEATURE_DIR, info['feature_file'])
        if not os.path.exists(feat_path):
            continue
            
        # 정답(Ground Truth) 맵핑 적용
        ref_sequence = [mapping_dict.get(g, g) for g in info['gloss_sequence']]
        
        # 모델 예측
        feature_tensor = torch.tensor(np.load(feat_path), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(feature_tensor)
            predictions = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
            
        hyp_sequence = ctc_greedy_decode(predictions, idx_to_gloss, blank_idx=0)
        
        # --- 지표 계산 ---
        # 1. Exact Match
        if ref_sequence == hyp_sequence:
            exact_match_count += 1
            
        # 2. WER (편집 거리)
        dist, ref_len = calculate_wer(ref_sequence, hyp_sequence)
        total_edit_distance += dist
        total_ref_words += ref_len
        
        # 3. 토큰 F1 Score
        p, r, f1 = calculate_token_f1(ref_sequence, hyp_sequence)
        sum_precision += p
        sum_recall += r
        sum_f1 += f1
        
        # (선택) 처음 3개 샘플만 어떻게 예측했는지 터미널에 출력해줌
        if i < 3:
            print(f"\n[Sample {i+1}]")
            print(f"정답 (Ref): {ref_sequence}")
            print(f"예측 (Hyp): {hyp_sequence}")
            print(f"-> WER: {dist/max(1, ref_len):.2f} | F1: {f1:.2f}")

    # ==========================================
    # 3. 최종 결과 리포팅
    # ==========================================
    seq_accuracy = (exact_match_count / total_samples) * 100
    overall_wer = (total_edit_distance / total_ref_words) * 100 if total_ref_words > 0 else 0
    
    avg_precision = (sum_precision / total_samples) * 100
    avg_recall = (sum_recall / total_samples) * 100
    avg_f1 = (sum_f1 / total_samples) * 100

    print("\n" + "="*50)
    print("🎉 [최종 Test Set 평가 결과 보고서] 🎉")
    print("="*50)
    print(f"1. Sequence Accuracy (문장 완전 일치율) : {seq_accuracy:.2f}%")
    print(f"2. WER (단어 오류율, 낮을수록 좋음)     : {overall_wer:.2f}%")
    print(f"3. Token Precision (정밀도)             : {avg_precision:.2f}%")
    print(f"4. Token Recall (재현율)                : {avg_recall:.2f}%")
    print(f"5. Token F1-Score (조화 평균)           : {avg_f1:.2f}%")
    print("="*50)
    print("💡 해석 가이드:")
    print("- WER이 40~50% 이하라면 베이스라인으로서 매우 훌륭한 상태입니다.")
    print("- Sequence Accuracy는 매우 깐깐한 지표라 10%대만 나와도 정상입니다.")
    print("- 이 결과를 2단계 LLM 파인튜닝의 입력으로 사용하시면 됩니다.")

if __name__ == "__main__":
    evaluate_model()