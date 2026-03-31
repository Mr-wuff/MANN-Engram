import torch
import torch.nn as nn
import torch.nn.functional as F

class BackboneNet(nn.Module):
    """Base feed-forward backbone for the routing engine."""
    def __init__(self, d=1152, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), 
            nn.GELU(), 
            nn.Dropout(0.1), 
            nn.Linear(hidden, hidden)
        )
        
    def forward(self, x): 
        return self.net(x)


class SkewGaussian(nn.Module):
    """
    Skew-Gaussian Distribution Estimator.
    Calculates Mahalanobis distance in latent space for multimodal semantic routing.
    """
    def __init__(self, d=1152, rank=16, hidden=256):
        super().__init__()
        self.backbone = BackboneNet(d, hidden)
        self.head_d = nn.Linear(hidden, d)
        self.head_L = nn.Linear(hidden, d * rank)
        self.head_delta = nn.Linear(hidden, d)
        self.head_alpha = nn.Linear(hidden, d)

    def forward(self, q):
        h = self.backbone(q)
        return self.head_d(h), self.head_L(h), self.head_delta(h), self.head_alpha(h)

    def compute_score(self, q, c):
        log_d, L_flat, delta, alpha = self.forward(q)
        log_d = torch.clamp(log_d, min=-15.0, max=15.0) 
        
        B, D = q.shape
        R = L_flat.shape[-1] // D
        L = L_flat.view(B, D, R)
        center = q + delta
        diff = c - center.unsqueeze(1)
        
        d_val = torch.exp(log_d)
        d_inv = 1.0 / d_val
        mahal = torch.sum(diff**2 * d_inv.unsqueeze(1), dim=-1)
        
        DinvL = d_inv.unsqueeze(-1) * L
        M = torch.eye(R, device=q.device).unsqueeze(0) + torch.bmm(L.transpose(1, 2), DinvL)
        Mi = torch.linalg.inv(M + 1e-4 * torch.eye(R, device=q.device).unsqueeze(0)) 
        
        v = diff * d_inv.unsqueeze(1)
        w = torch.bmm(L.transpose(1, 2), v.transpose(1, 2))
        mahal = mahal - torch.sum(w * torch.bmm(Mi, w), dim=1)
        
        ld = torch.sum(log_d, dim=-1, keepdim=True) + torch.linalg.slogdet(M)[1].unsqueeze(-1)
        lp = -0.5 * (mahal + ld) + F.logsigmoid(torch.sum(alpha.unsqueeze(1) * diff, dim=-1))
        
        return lp