import torch
import torch.nn as nn
import math


class PatchTST(nn.Module):
    def __init__(
        self,
        hist_len,
        pred_len,
        in_dim,
        city_num,
        batch_size,
        device,
        patch_len=4,
        stride=2,
        d_model=64,
        nhead=4,
        num_layers=2
    ):
        super(PatchTST, self).__init__()

        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size
        self.in_dim = in_dim

        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        self.num_patches = (hist_len - patch_len) // stride + 1

        self.patch_embedding = nn.Linear(patch_len * in_dim, d_model)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.head = nn.Linear(d_model, pred_len)

    def forward(self, pm25_hist, feature):

        B, T, N, _ = pm25_hist.shape

        x = torch.cat(
            [pm25_hist, feature[:, :self.hist_len]],
            dim=-1
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B * N, T, self.in_dim)

        patches = []
        for i in range(0, T - self.patch_len + 1, self.stride):
            patch = x[:, i:i + self.patch_len, :]
            patch = patch.reshape(B * N, -1)
            patches.append(patch)

        patches = torch.stack(patches, dim=1)
        patches = self.patch_embedding(patches)
        patches = patches + self.pos_embedding

        encoded = self.encoder(patches)

        pooled = encoded.mean(dim=1)

        pm25_pred = self.head(pooled)
        pm25_pred = pm25_pred.view(B, N, self.pred_len, 1)
        pm25_pred = pm25_pred.permute(0, 2, 1, 3)

        return pm25_pred
