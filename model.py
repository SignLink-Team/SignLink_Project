import torch
import torch.nn as nn

class SignLanguageModel(nn.Module):
    # 여기에 dropout=0.5 매개변수가 추가되었습니다!
    def __init__(self, input_dim=255, hidden_dim=256, num_layers=3, num_classes=100, dropout=0.5):
        super(SignLanguageModel, self).__init__()
        
        # LSTM 자체에 드롭아웃 적용
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout 
        )
        
        # 분류기(FC Layer)로 넘어가기 전에도 드롭아웃 적용
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out)
        logits = self.fc(out)
        return logits