import torch.nn as nn
import torch
from utils.dct import dct,idct

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channel_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

        if channel_first:
            self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
            self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FreqMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., 
                 channel_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.channel_first = channel_first

        if channel_first:
            self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
            self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        if not self.channel_first:
            b, f, _ = x.shape
            x = dct.dct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        else:
            b, c, t, j = x.shape
            x = x.permute(0, 3, 2, 1)  # [B,J,T,C]
            x = dct.dct(x).permute(0, 3, 2, 1).contiguous()  # [B,C,T,J]
            
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        if not self.channel_first:
            x = dct.idct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        else:
            x = x.permute(0, 3, 2, 1)  # [B,J,T,C]
            x = dct.idct(x).permute(0, 3, 2, 1).contiguous()  # [B,C,T,J]
            
        return x