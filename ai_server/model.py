import torch
import torch.nn as nn

class SignLanguageModel(nn.Module):
    def __init__(self, input_dim=783, hidden_dim=512, num_classes=500, num_layers=2, dropout=0.3):
        super(SignLanguageModel, self).__init__()
        
        # 1D-CNN (Temporal Subsampling & Local Feature Extraction)
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # BiLSTM (Global Context Extraction)
        self.lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim // 2, 
            num_layers=num_layers,
            bidirectional=True, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classifier
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # (Batch, Seq, Features) -> (Batch, Features, Seq)
        x = x.transpose(1, 2)  
        x = self.conv1d(x)
        
        # (Batch, Features, Seq) -> (Batch, Seq, Features)
        x = x.transpose(1, 2)  
        
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        
        logits = self.fc(out) 
        return logits