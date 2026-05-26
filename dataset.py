import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm

def stratified_sentence_split(info_list, mapping_dict, train_ratio=0.8, val_ratio=0.1):
    """
    다중 라벨(수어 문장)의 층화 추출을 수행합니다.
    빈도수가 낮은 단어가 Train, Val, Test에 골고루 들어가도록 보장합니다.
    """
    # 1. 매핑된 단어들의 문서 빈도수(DF) 계산
    word_freq = Counter()
    for info in info_list:
        mapped_words = set([mapping_dict.get(g, g) for g in info['gloss_sequence']])
        word_freq.update(mapped_words)

    # 2. 문장의 희소성 점수(문장 내 가장 희귀한 단어의 빈도수) 계산 후 정렬
    def get_rarest_freq(info):
        mapped = set([mapping_dict.get(g, g) for g in info['gloss_sequence']])
        if not mapped: return float('inf')
        return min([word_freq[w] for w in mapped])

    # 희귀한 단어를 가진 문장부터 먼저 배치하기 위해 오름차순 정렬
    sorted_info = sorted(info_list, key=get_rarest_freq)

    train_list, val_list, test_list = [], [], []
    split_counts = {'train': Counter(), 'val': Counter(), 'test': Counter()}

    # 3. 탐욕적(Greedy) 분배
    for info in sorted_info:
        mapped = set([mapping_dict.get(g, g) for g in info['gloss_sequence']])
        if not mapped:
            train_list.append(info)
            continue
            
        # 이 문장에서 가장 희귀한 단어를 찾음
        rarest_word = min(mapped, key=lambda w: word_freq[w])
        
        t_c = split_counts['train'][rarest_word]
        v_c = split_counts['val'][rarest_word]
        te_c = split_counts['test'][rarest_word]
        
        # 필수 할당 규칙: Train(1순위) -> Val(2순위) -> Test(3순위)
        if t_c == 0:
            target = 'train'
        elif v_c == 0:
            target = 'val'
        elif te_c == 0:
            target = 'test'
        else:
            # 3곳 모두 최소 1개씩 있다면, 목표 비율에 맞춰 배정
            total_expected_ratio = train_ratio / val_ratio
            if t_c / max(1, v_c) <= total_expected_ratio and t_c / max(1, te_c) <= total_expected_ratio:
                target = 'train'
            elif v_c <= te_c:
                target = 'val'
            else:
                target = 'test'

        # 결정된 target에 리스트 및 카운트 업데이트
        if target == 'train':
            train_list.append(info)
            for w in mapped: split_counts['train'][w] += 1
        elif target == 'val':
            val_list.append(info)
            for w in mapped: split_counts['val'][w] += 1
        else:
            test_list.append(info)
            for w in mapped: split_counts['test'][w] += 1

    print(f"✅ Split 완료: Train {len(train_list)} | Val {len(val_list)} | Test {len(test_list)}")
    return train_list, val_list, test_list


class SignLanguageInMemoryDataset(Dataset):
    def __init__(self, data_list, feature_dir, gloss_to_idx, mapping_dict):
        self.data_cache = []
        
        for info in tqdm(data_list, desc="Loading to RAM"):
            feat_path = os.path.join(feature_dir, info['feature_file'])
            if not os.path.exists(feat_path):
                continue
                
            feature_tensor = torch.tensor(np.load(feat_path), dtype=torch.float32)
            
            target_indices = []
            for gloss in info['gloss_sequence']:
                mapped_gloss = mapping_dict.get(gloss, gloss)
                # 사전에 없을 시 <UNK> 인덱스 가져오기 (이제 <UNK>가 무조건 존재함)
                idx = gloss_to_idx.get(mapped_gloss, gloss_to_idx.get("<UNK>"))
                target_indices.append(idx)
                
            target_tensor = torch.tensor(target_indices, dtype=torch.long)
            self.data_cache.append((feature_tensor, target_tensor))

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        return self.data_cache[idx]


def ctc_collate_fn(batch):
    features, targets = zip(*batch)
    
    feat_lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)
    tgt_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)
    
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
    # 🌟 [수정됨] 정답 패딩 값을 0에서 -1로 변경! (blank 인덱스인 0과 혼동 방지)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=-1)
    
    return padded_features, padded_targets, feat_lengths, tgt_lengths