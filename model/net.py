import torch
import torch.nn as nn
from torchcrf import CRF


class LSTM_CRF_Model(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=200, num_tags=2, num_layers=4, dropout=0.5):
        super(LSTM_CRF_Model, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, embeds):
        # embeds: (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeds)
        # lstm_out: (batch_size, seq_len, hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        # lstm_feats: (batch_size, seq_len, num_tags)
        return lstm_feats

    def loss(self, feats, tags, mask=None):
        # tags: (batch_size, seq_len)
        # mask: (batch_size, seq_len)
        return -self.crf(feats, tags, mask)

    def predict(self, feats, mask=None):
        return self.crf.decode(feats, mask)
