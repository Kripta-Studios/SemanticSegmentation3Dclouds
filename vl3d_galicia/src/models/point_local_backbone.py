import torch
import torch.nn as nn
import torch.nn.functional as F

def get_knn_indices(pos, k=16):
    """
    Computes KNN indices for a batch of point clouds.
    Args:
        pos: [B, N, 3] coordinates
        k: number of nearest neighbors
    Returns:
        idx: [B, N, k] indices of nearest neighbors
    """
    # pos: [B, N, 3]
    # pairwise distances
    dist = torch.cdist(pos, pos) # [B, N, N]
    # get top k
    # set diagonal to large number to avoid self? Or keep self? 
    # Usually in EdgeConv keeping self is okay, but let's keep self.
    _, idx = torch.topk(dist, k=k, dim=-1, largest=False)
    return idx

def get_graph_feature(x, k=16, idx=None, pos=None):
    """
    Args:
        x: [B, N, C]
        k: number of nearest neighbors
        idx: [B, N, k] indices if precomputed
        pos: [B, N, 3] coordinates to compute KNN if idx is None
    Returns:
        feature: [B, N, k, 2*C] (x_i, x_j - x_i)
    """
    batch_size, num_points, num_dims = x.size()
    
    if idx is None:
        idx = get_knn_indices(pos, k=k)
    
    # x is [B, N, C]
    # we want to index x to get [B, N, k, C]
    
    # idx is [B, N, k]
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx_flat = (idx + idx_base).view(-1)
    
    x_flat = x.view(batch_size * num_points, -1)
    x_neighbors = x_flat[idx_flat].view(batch_size, num_points, k, num_dims) # [B, N, k, C]
    
    x_expanded = x.unsqueeze(2).expand(batch_size, num_points, k, num_dims)
    
    # return [B, N, k, 2C] -> EdgeConv style: (x_i, x_j - x_i)
    feature = torch.cat((x_expanded, x_neighbors - x_expanded), dim=3)
    return feature

class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x, pos, knn_idx=None):
        # x: [B, N, C]
        # pos: [B, N, 3]
        if knn_idx is None:
            knn_idx = get_knn_indices(pos, self.k)
        
        # feature: [B, N, k, 2C]
        feat = get_graph_feature(x, k=self.k, idx=knn_idx)
        
        # pass through MLP
        B, N, K, C2 = feat.size()
        feat = feat.view(B*N*K, C2)
        feat = self.mlp(feat)
        feat = feat.view(B, N, K, -1)
        
        # aggregate neighbors via max pooling
        feat = feat.max(dim=2, keepdim=False)[0] # [B, N, out_channels]
        return feat

class PointLocalBackbone(nn.Module):
    def __init__(self, in_channels, out_classes=7, k=16, jepa_dim=0):
        super().__init__()
        self.k = k
        
        # If we concatenate a JEPA embedding, it adds to the input channels.
        # But JEPA embedding is typically [B, D]. We will broadcast it to [B, N, D]
        # in the forward pass before passing to this network.
        # So in_channels already includes JEPA dim if passed.
        
        # 1. Feature extraction per point
        self.fc1 = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 2. Local aggregation 1
        self.edge1 = EdgeConvBlock(64, 128, k=k)
        
        # 3. Local aggregation 2
        self.edge2 = EdgeConvBlock(128, 256, k=k)
        
        # 4. Global Context
        self.fc_global = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        # 5. Segmentation Head
        # Concat features: fc1 (64) + edge1 (128) + edge2 (256) + global (512) = 960
        self.seg_head = nn.Sequential(
            nn.Linear(960, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, out_classes)
        )
        
    def forward(self, x, coords, jepa_emb=None):
        """
        x: [B, N, C]
        coords: [B, N, 3] (for computing distances)
        jepa_emb: [B, D] (optional)
        """
        B, N, C = x.size()
        
        if jepa_emb is not None:
            # Broadcast JEPA to [B, N, D]
            D = jepa_emb.size(1)
            jepa_exp = jepa_emb.unsqueeze(1).expand(B, N, D)
            x = torch.cat([x, jepa_exp], dim=-1)
            
        # Compute fixed KNN indices once based on coords
        knn_idx = get_knn_indices(coords, k=self.k)
        
        # fc1
        x_flat = x.view(B*N, -1)
        f1 = self.fc1(x_flat).view(B, N, -1) # [B, N, 64]
        
        # edge1
        f2 = self.edge1(f1, coords, knn_idx) # [B, N, 128]
        
        # edge2
        f3 = self.edge2(f2, coords, knn_idx) # [B, N, 256]
        
        # global
        f_global_pts = self.fc_global(f3.view(B*N, -1)).view(B, N, -1)
        f_global = f_global_pts.max(dim=1, keepdim=True)[0] # [B, 1, 512]
        f_global = f_global.expand(B, N, 512) # [B, N, 512]
        
        # concat
        concat_feat = torch.cat([f1, f2, f3, f_global], dim=-1) # [B, N, 960]
        
        # seg head
        out = self.seg_head(concat_feat.view(B*N, -1))
        out = out.view(B, N, -1) # [B, N, out_classes]
        
        return out
