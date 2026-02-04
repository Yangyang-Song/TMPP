import torch
import torch.nn as nn
import math

class Fixed_Memory(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device):
        super(TMPP, self).__init__()
        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.hid_dim = 32
        self.out_dim = 1
        self.nhead = 4
        self.num_layers = 2

        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)

        self.position_encoding = self._generate_position_encoding()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hid_dim,
            nhead=self.nhead,
            dim_feedforward=self.hid_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

        self.fc_out_direct = nn.Linear(self.hid_dim, self.pred_len)

    def _generate_position_encoding(self, max_len=50000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hid_dim, 2) * (-math.log(10000.0) / self.hid_dim))
        pe = torch.zeros(max_len, self.hid_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.to(self.device)

    def forward(self, pm25_hist, feature):

        memory = torch.zeros(
            self.batch_size * self.city_num,
            self.hist_len,
            self.hid_dim
        ).to(self.device)

        for t in range(self.hist_len):
            x = torch.cat((pm25_hist[:, t], feature[:, t]), dim=-1)
            x = self.fc_in(x)
            x = x.view(self.batch_size * self.city_num, self.hid_dim) 
            memory[:, t] = x

        pe = self.position_encoding[:self.hist_len]
        pe = pe.unsqueeze(0).expand(memory.size(0), -1, -1)
        memory = memory + pe

        encoded = self.transformer_encoder(memory)

        hn = encoded[:, -1]

        out = self.fc_out_direct(hn)
        out = out.view(self.batch_size, self.city_num, self.pred_len)
        out = out.permute(0, 2, 1)
        out = out.unsqueeze(-1)

        return out