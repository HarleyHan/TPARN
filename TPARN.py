"""
    unoffical implementation of 
    TPARN: Triple-Path Attentive Recurrent Network for Time-Domain Multichannel Speech Enhancement, ICASSP 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# input: (P, N)
# frames: (P, T=num_frames, L=frame_len)
# chunks: (P, C=num_chunks, R=chunk_size, L)
# linear: (P, C, R, L) -> (P, C, R, D)

# 1st tparn: (P, C, R, D) -> (P, C, R, D)
# 2nd tparn: (P, C, R, 2*D) -> (P, C, R, D)
# 3rd tparn: (P, C, R, 3*D) -> (P, C, R, D)
# 4th tparn: (P, C, R, 4*D) -> (P, C, R, D)

# tparn: 
#   (P, C, R, k*D) -> (P, C, R, D) 
#   -> (P*C, R, D) -> (P*R, C, D) -> (R*C, P, D) -> (P, C, R, D)

# arn: rnn block + attention block + feedfoward block


class RNN_block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.layernorm1 = nn.LayerNorm(channels)
        self.layernorm2 = nn.LayerNorm(channels)
        
        self.rnn = nn.LSTM(channels, channels, batch_first=True, bidirectional=True)
        
        self.linear = nn.Linear(3*channels, channels)
        
    def forward(self, x):
        # x: (B, T, D)
        
        # (B, T, D)
        y = self.layernorm1(x)
        
        # (B, T, 2*D)
        y, _ = self.rnn(y)
        
        # (B, T, D)
        z = self.layernorm2(x)
        
        # (B, T, 3*D)
        z = torch.cat([y, z], dim=-1)
        
        # (B, T, D)
        z = self.linear(z)
        
        return z
    
    
class Att_block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.layernorm1 = nn.LayerNorm(channels)
        self.layernorm2 = nn.LayerNorm(channels)
        
        self.q = nn.Parameter(torch.randn(1, channels))
        self.k = nn.Parameter(torch.randn(1, channels))
        self.v = nn.Parameter(torch.randn(1, channels))
        
        self.linear_q = nn.Linear(channels, channels)
        
        self.linear_v1 = nn.Linear(channels, channels)
        self.linear_v2 = nn.Linear(channels, channels)
        
        self.channels = channels
        
    def forward(self, x):
        # (B, T, D)
        query = self.layernorm1(x)
        key = self.layernorm2(x)
        value = key.clone()
        
        # (B, T, D) 
        k = key * torch.sigmoid(self.k).unsqueeze(0)
        q = self.linear_q(query) * torch.sigmoid(self.q)
        v = value * torch.sigmoid(self.linear_v1(self.v)).unsqueeze(0) * torch.tanh(self.linear_v2(self.v)).unsqueeze(0)
        
        # (B, T, T) 
        score = torch.softmax(torch.matmul(q, k.transpose(1, 2))/(self.channels**0.5), dim=-1)
        
        # (B, T, D)
        out = torch.matmul(score, v)
        
        return q + out
    
    
class Feedforward_block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.layernorm1 = nn.LayerNorm(channels)
        self.layernorm2 = nn.LayerNorm(channels)
        
        self.linear = nn.Sequential(
            nn.Linear(channels, 4*channels),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(4*channels, channels)
        )
        
    def forward(self, x):
        # x: (B, T, D)
        
        # (B, T, D)
        y = self.layernorm1(x)
        
        # (B, T, D)
        y = self.linear(y)
        
        # (B, T, D)
        z = self.layernorm2(x)
        
        return y + z
    


class ARN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.rnn = RNN_block(channels)
        
        self.att = Att_block(channels)
        
        self.ff = Feedforward_block(channels)
        
    def forward(self, x):
        # x: (B, T, D)
        
        # (B, T, D)
        y = self.rnn(x)
        
        # (B, T, D)
        y = self.att(y)
        
        # (B, T, D)
        y = self.ff(y)
        
        return y
        
        
        
class TPARN_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.linear = nn.Linear(in_channels, out_channels)
        
        self.intra_chunk_arn = ARN(out_channels)
        self.inter_chunk_arn = ARN(out_channels)
        self.inter_channel_arn = ARN(out_channels)

    def forward(self, x: torch.Tensor):
        # x: (B, P, C, R, k*D)
        
        # (B, P, C, R, D)
        y = self.linear(x)
        
        B, P, C, R, D = y.shape
        
        # (B*P*C, R, D)
        y = y.flatten(0, 2)
        
        # (B*P*C, R, D)
        y = self.intra_chunk_arn(y)
        
        # (B*P*R, C, D)
        y = y.reshape(B, P, C, R, D).permute(0, 1, 3, 2, -1).flatten(0, 2)
        
        # (B*P*R, C, D)
        y = self.inter_chunk_arn(y)
        
        # (B*R*C, P, D)
        y = y.reshape(B, P, R, C, D).permute(0, 2, 3, 1, -1).flatten(0, 2)
        
        # (B*R*C, P, D)
        y = self.inter_channel_arn(y)
        
        # (B, P, C, R, D)
        y = y.reshape(B, R, C, P, D).permute(0, 3, 2, 1, -1)
        
        return y
    
    
class TPARN(nn.Module):
    """
        P: num_channels
        D: num_features
        L: frame_size
        K: frame_shift
        R: chunk_size
        S: chunk_shift
    """
    def __init__(self, P=8, D=256, L=32, K=16, R=252, S=126):
        super().__init__()
        
        self.K = K
        self.S = S
        
        self.kernel_frame = nn.Parameter(torch.eye(L).unsqueeze(1), requires_grad=False)
        self.kernel_chunk = nn.Parameter(torch.eye(R).unsqueeze(1), requires_grad=False)
        
        self.linear_in = nn.Linear(L, D)
        self.linear_out = nn.Linear(D, L)
        
        self.tparn_blocks = nn.ModuleList()
        for k in range(2):
            self.tparn_blocks.append(
                TPARN_block((k+1) * D, D)
            )     
        
    def forward(self, x: torch.Tensor):
        # x: (B, P, N)
        B, P, _ = x.shape
        input_shape = x.shape
        
        # (B*P, 1, N)
        x = x.flatten(0, 1).unsqueeze(1)

        
        
        # (B*P, L, T)
        x = F.conv1d(x, weight=self.kernel_frame, stride=self.K, padding=1)

        
        
        # (B*P*L, 1, T)
        x = x.flatten(0, 1).unsqueeze(1)

        
        
        # (B*P*L, R, C)
        x = F.conv1d(x, weight=self.kernel_chunk, stride=self.S, padding=1)
        
        # (B, P, L, R, C) -> (B, P, C, R, L)
        x = x.reshape(B, P, -1, *x.shape[1:]).permute(0, 1, 4, 3, 2)
        
        # x: (B, P, C, R, D)
        x = self.linear_in(x)
        
        for k in range(len(self.tparn_blocks)):
            y = self.tparn_blocks[k](x)
            # x: (B, P, C, R, k*D)
            x = torch.concat([x, y], dim=-1)
        
        # x: (B, P, C, R, L)
        y = self.linear_out(y)
        
        # (B*P*L, R, C)
        y = y.permute(0, 1, 4, 3, 2).flatten(0, 2)
        
        # (B*P*L, 1, T)
        y = F.conv_transpose1d(y, weight=self.kernel_chunk, stride=self.S, padding=1)
        
        # (B*P, L, T)
        y = y.reshape(B*P, -1, y.shape[-1])
        
        # (B*P, 1, N)
        y = F.conv_transpose1d(y, weight=self.kernel_frame, stride=self.K, padding=1)
        
        # (B, P, N)
        y = y.reshape(B, P, -1)

        if y.shape[-1] != input_shape[-1]:
            # 在最后补零
            pad_size = input_shape[-1] - y.shape[-1]
            y = F.pad(y, (0, pad_size))  # (左，右)

        est = {
             "wav": y,
         }
            
        return est
    
    
    
if __name__ == "__main__":
    

    layer = TPARN()
    x = torch.rand(1, 8, 16000*4)  # original input [B, C, length]
    print("input shape:", x.shape)
    
    # Get the output after the forward pass
    output = layer(x)
    print("output shape:", output.shape)
    
    total_num = sum(p.numel() for p in layer.parameters())
    print("Total parameters:", total_num)
    
            
