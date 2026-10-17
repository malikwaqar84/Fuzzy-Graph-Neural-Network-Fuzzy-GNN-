"""The Fuzzy-GNN of Section 3 (Eqs. 8-16) and its ablation variants."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def inv_softplus(x: float) -> float:
    return math.log(math.expm1(x))


class FuzzyAttentionLayer(nn.Module):
    """Eqs. 13-14: multi-head attention combined with fuzzy edge strength, renormalised."""

    def __init__(self, d_in, hidden, heads, dropout, use_attention=True):
        super().__init__()
        self.W = nn.Linear(d_in, hidden * heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, hidden))
        self.a_dst = nn.Parameter(torch.empty(heads, hidden))
        nn.init.xavier_uniform_(self.a_src); nn.init.xavier_uniform_(self.a_dst)
        self.heads, self.hidden, self.use_attention = heads, hidden, use_attention
        self.drop = nn.Dropout(dropout)

    def forward(self, h, mu):                       # h [M, C, d], mu [M, C, C]
        M, C, _ = h.shape
        Wh = self.W(h).view(M, C, self.heads, self.hidden)            # [M, C, H, k]
        if self.use_attention:
            e_src = (Wh * self.a_src).sum(-1)                          # [M, C, H]
            e_dst = (Wh * self.a_dst).sum(-1)
            e = F.leaky_relu(e_src.unsqueeze(2) + e_dst.unsqueeze(1), 0.2)   # [M, i, j, H]
            alpha = torch.softmax(e, dim=2)
        else:
            alpha = torch.ones(M, C, C, self.heads, device=h.device) / C
        w = alpha * mu.unsqueeze(-1)
        w = w / (w.sum(dim=2, keepdim=True) + 1e-8)                   # renormalisation in Eq. 14
        w = self.drop(w)
        out = torch.einsum("mijh,mjhk->mihk", w, Wh)
        self.last_alpha = alpha.detach()
        return F.elu(out.reshape(M, C, self.heads * self.hidden))


class GCNLayer(nn.Module):
    """Symmetric-normalised graph convolution, used only by the hard-threshold ablation."""

    def __init__(self, d_in, d_out, dropout):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, A):
        deg = A.sum(-1)
        Dm = deg.clamp(min=1e-6).pow(-0.5)
        An = Dm.unsqueeze(-1) * A * Dm.unsqueeze(-2)
        return F.elu(self.drop(torch.bmm(An, self.lin(h))))


class FuzzyGNN(nn.Module):
    needs_raw = False

    def __init__(self, cfg, n_channels, d_short, d_long):
        super().__init__()
        m = cfg["model"]
        self.mode, self.learn = m["membership"], m["learn_membership"]
        self.use_eai, self.multiscale = m["use_eai"], m["multiscale"]
        self.tau = m["hard_tau"]
        self.d_short = d_short
        th = torch.tensor(m["theta_init"])
        if self.mode == "gaussian":
            sp = torch.tensor(inv_softplus(m["gaussian_sigma"]))
        else:
            sp = torch.tensor(inv_softplus(m["kappa_init"]))
        if self.learn:
            self.theta, self.spread_raw = nn.Parameter(th), nn.Parameter(sp)
        else:
            self.register_buffer("theta", th); self.register_buffer("spread_raw", sp)
        self.register_buffer("static_c", torch.zeros(n_channels, n_channels))
        d_in = d_short + (d_long if self.multiscale else 0) + (1 if self.use_eai else 0)
        H, k = m["heads"], m["hidden"]
        if self.mode == "hard_gcn":
            dims = [d_in] + [H * k] * m["layers"]
            self.layers = nn.ModuleList([GCNLayer(dims[i], dims[i + 1], m["dropout"]) for i in range(m["layers"])])
        else:
            dims = [d_in] + [H * k] * m["layers"]
            self.layers = nn.ModuleList([FuzzyAttentionLayer(dims[i], k, H, m["dropout"], m["use_attention"])
                                         for i in range(m["layers"])])
        self.q = nn.Linear(H * k, 1, bias=False)          # Eq. 15
        self.out = nn.Linear(H * k, 1)                    # Eq. 16

    # --------------------------------------------------------------
    def membership(self, c):
        """Eq. 10 (and the ablation alternatives). c [M, C, C] -> mu [M, C, C]."""
        a = c.abs()
        if self.mode == "static":
            a = self.static_c.unsqueeze(0).expand_as(a)
        if self.mode == "hard_gcn":
            mu = (a > self.tau).float()
        elif self.mode == "gaussian":
            sigma = F.softplus(self.spread_raw)
            mu = torch.exp(-(a - self.theta) ** 2 / (2 * sigma ** 2))
        else:  # sigmoid (proposed) or static
            kappa = F.softplus(self.spread_raw)
            mu = torch.sigmoid(kappa * (a - self.theta))
        eye = torch.eye(c.shape[-1], device=c.device).unsqueeze(0)
        return mu * (1 - eye) + eye

    @staticmethod
    def eai(mu, act):
        """Eq. 12: fuzzy-weighted mean activation of the neighbours. act [M, C]."""
        C = mu.shape[-1]
        off = mu * (1 - torch.eye(C, device=mu.device)).unsqueeze(0)
        return (off * act.unsqueeze(1)).sum(-1) / (off.sum(-1) + 1e-8)

    def node_inputs(self, batch):
        B, T, C, D = batch["feat"].shape
        x = batch["feat"].reshape(B * T, C, D)
        c = batch["corr"].reshape(B * T, C, C)
        act = batch["act"].reshape(B * T, C)
        mu = self.membership(c)
        parts = [x[..., :self.d_short]]
        if self.multiscale:
            parts.append(x[..., self.d_short:])
        e = self.eai(mu, act)
        if self.use_eai:
            parts.append(e.unsqueeze(-1))
        return torch.cat(parts, -1), mu, e

    def forward(self, batch, return_eai=False):
        B, T = batch["feat"].shape[:2]
        h, mu, e = self.node_inputs(batch)
        for layer in self.layers:
            h = layer(h, mu)
        beta = torch.softmax(self.q(h).squeeze(-1), dim=-1)          # Eq. 15
        Z = (beta.unsqueeze(-1) * h).sum(1)
        logit = self.out(Z).squeeze(-1).view(B, T)                   # Eq. 16 (before sigmoid)
        if return_eai:
            return logit, e.view(B, T, -1)
        return logit

    def membership_params(self):
        return {"theta": float(self.theta.detach()), "kappa_or_sigma": float(F.softplus(self.spread_raw.detach()))}
