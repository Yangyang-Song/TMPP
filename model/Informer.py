import torch
import torch.nn as nn
import math
import random

class ProbAttention(nn.Module):
    def __init__(self, factor=5, dropout=0.1):
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        sample_k = min(self.factor * int(math.log(L_K)), L_K)
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)

        K_sample = K[:, :, index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        M = Q_K_sample.max(-1)[0] - Q_K_sample.mean(-1)
        top_k = min(self.factor * int(math.log(L_Q)), L_Q)
        top_indices = M.topk(top_k, sorted=False)[1]

        Q_reduced = Q.gather(
            2, top_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        )

        scores = torch.matmul(Q_reduced, K.transpose(-2, -1)) / math.sqrt(D)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)

        context_full = V.mean(dim=-2, keepdim=True).expand(B, H, L_Q, D)
        context_full.scatter_(
            2,
            top_indices.unsqueeze(-1).expand(-1, -1, -1, D),
            context
        )

        return context_full



class InformerAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_head = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn = ProbAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape

        Q = self.q_proj(x).view(B, L, self.nhead, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.nhead, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.nhead, self.d_head).transpose(1, 2)

        context = self.attn(Q, K, V)
        context = context.transpose(1, 2).contiguous().view(B, L, D)

        return self.out_proj(context)


class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = InformerAttentionLayer(d_model, nhead, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(x))
        x = self.norm1(x)
        x = x + self.dropout(self.ffn(x))
        x = self.norm2(x)
        return x


class Informer(nn.Module):
    def __init__(
        self,
        hist_len,
        pred_len,
        in_dim,
        city_num,
        batch_size,
        device,
        d_model=64,
        nhead=4,
        num_layers=2
    ):
        super().__init__()

        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size
        self.in_dim = in_dim

        self.embedding = nn.Linear(in_dim, d_model)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, hist_len, d_model)
        )

        self.encoder = nn.ModuleList([
            InformerEncoderLayer(d_model, nhead)
            for _ in range(num_layers)
        ])

        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, pm25_hist, feature):

        B, T, N, _ = pm25_hist.shape

        x = torch.cat(
            [pm25_hist, feature[:, :self.hist_len]],
            dim=-1
        )

        # reshape to (B*N, T, D)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B * N, T, self.in_dim)

        x = self.embedding(x)
        x = x + self.pos_embedding[:, :T]

        for layer in self.encoder:
            x = layer(x)

        x = x.mean(dim=1)
        pm25_pred = self.projection(x)

        pm25_pred = pm25_pred.view(B, N, self.pred_len, 1)
        pm25_pred = pm25_pred.permute(0, 2, 1, 3)

        return pm25_pred
