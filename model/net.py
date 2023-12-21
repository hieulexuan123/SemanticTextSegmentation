import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=768, hidden_size=params['hidden_dim'], num_layers=params['num_layers'],
                            dropout=params['dropout'], bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * params['hidden_dim'], 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        fc_out = self.fc(out.reshape(-1, out.shape[2]))
        return F.log_softmax(fc_out, dim=1) #batch_size*seq_len x 2

