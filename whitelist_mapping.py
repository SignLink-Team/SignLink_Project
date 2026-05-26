import json

# 1. 절대 지우면 안 되는 의료 필수 고유명사 (Whitelist)
# (이전에 뽑아드린 리스트 + 새로 발견된 단어들 모두 포함)
MEDICAL_WHITELIST = {
    "결막염", "방광암", "위암", "뇌염", "췌장염", "충수염", "류마티스",
    "정맥류", "동맥경화", "각화증", "결석", "아토피", "고혈압", "녹내장",
    "중풍1", "심정지1", "치주염", "항응고제", "항생", "인슐린1", "요오드",
    "식이섬유소", "나트륨", "망막", "혈소판", "갑상선2", "항문외과", 
    "임플란트1", "라미네이트", "급성디스크", "고지혈약", "질정제", "엽산제", 
    "척추2", "코로나2", "소견서", "입원1", "급성", "전립선", "쇄골1",
    "비타민D", "칼륨", "미네랄", "동맥", "경화", "정맥", "요로"
}

def generate_unk_mapped_dict(freq_json_path, output_path, threshold=5):
    # 1. 빈도수 파일 로드
    with open(freq_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    gloss_freq = data['gloss_freq']
    
    # 2. 매핑 사전 생성 (원래 단어 -> 모델이 학습할 최종 단어)
    mapped_dict = {}
    unk_count = 0
    whitelist_saved_count = 0
    
    for gloss, freq in gloss_freq.items():
        if freq <= threshold:
            if gloss in MEDICAL_WHITELIST:
                # 빈도수가 낮아도 화이트리스트에 있으면 원본 유지
                mapped_dict[gloss] = gloss
                whitelist_saved_count += 1
            else:
                # 화이트리스트에 없는 꼬리 단어는 <UNK> 처리
                mapped_dict[gloss] = "<UNK>"
                unk_count += 1
        else:
            # 빈도수가 threshold보다 높은 일반 단어들은 그대로 유지
            mapped_dict[gloss] = gloss
            
    # 결과 요약 출력
    print(f"전체 단어 종류: {len(gloss_freq)}개")
    print(f"-> <UNK>로 치환되어 묶인 꼬리 단어: {unk_count}개")
    print(f"-> 빈도수가 낮지만 화이트리스트로 방어해낸 고유명사: {whitelist_saved_count}개")
    print(f"-> 딥러닝 모델이 맞춰야 할 최종 어휘(Vocabulary) 수: {len(set(mapped_dict.values()))}개")
    
    # 3. 생성된 매핑 사전을 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapped_dict, f, ensure_ascii=False, indent=4)
        
    return mapped_dict

# ==========================================
# 실행 예시
# ==========================================
if __name__ == "__main__":
    # gloss_freq_info.json 경로를 입력해 주세요.
    freq_file_path = "keypoint_data/gloss_freq_info.json"  
    output_mapping_path = "keypoint_data/gloss_mapping.json"
    
    mapping_dict = generate_unk_mapped_dict(freq_file_path, output_mapping_path)