import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplePointNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        # Point-wise MLPs
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, out_channels, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x shape: (B, N, C) -> transpose to (B, C, N)
        x = x.transpose(1, 2)
        
        # Point-wise features
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = self.bn3(self.conv3(x2))
        
        # Global feature via max pooling
        global_feat = torch.max(x3, 2, keepdim=True)[0]  # (B, out_channels, 1)
        
        # Expand global feature back to all points
        global_feat_expanded = global_feat.repeat(1, 1, x.shape[-1])
        
        # Concatenate point-wise and global features
        concat_feat = torch.cat([x3, global_feat_expanded], dim=1) # (B, 2 * out_channels, N)
        
        # Return to (B, N, 2*out_channels)
        return concat_feat.transpose(1, 2)
