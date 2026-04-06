import torch
import torch.nn as nn

class SignLanguageModel(nn.Module):
    # [수정] 데이터가 적으므로 num_layers를 2로 줄이고, dropout을 0.1로 대폭 낮췄습니다.
    def __init__(self, input_dim=255, hidden_dim=256, num_layers=2, num_classes=100, dropout=0.1):
        super(SignLanguageModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout 
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out)
        logits = self.fc(out)
        return logits