import torch.nn as nn
import torch


class RCMCL(nn.Module):
    def __init__(self, num_views, dims, num_classes, device):
        super(RCMCL, self).__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.EvidenceCollectors = nn.ModuleList([EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)])
        self.device = device

    def calculate_vacuity(self, alpha):
        """计算各视图的空虚度u = K / S"""
        vacuity_list = []
        for v in alpha:
            S = torch.sum(alpha[v], dim=1, keepdim=True)
            u = self.num_classes / S
            vacuity_list.append(u)
        # 将各视图的空虚度堆叠起来
        vacuity = torch.stack(vacuity_list, dim=1)
        return vacuity.squeeze(-1)

    def calculate_dissonance(self, evidences, device):
        dissonances = []
        for v in range(self.num_views):
            alpha_v = evidences[v] + 1
            S_v = torch.sum(alpha_v, dim=1, keepdim=True)
            b_v = (alpha_v - 1) / S_v
            K = b_v.shape[1]
            diss_v = torch.zeros(b_v.shape[0]).to(device)
            for k in range(K):
                # 提取b_j（j≠k）和b_k
                bj = b_v[:, torch.arange(K) != k]
                bk = b_v[:, k].unsqueeze(1)
                # 计算Bal(b_j, b_k)
                bal = torch.where(
                    (bj != 0) & (bk != 0),
                    1 - torch.abs(bj - bk + 1e-8) / (bj + bk + 1e-8),
                    torch.zeros_like(bj)
                )
                numerator = bk.squeeze(1) * torch.sum(bj * bal, dim=1)
                denominator = torch.sum(bj, dim=1)
                denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)  # 处理除零
                diss_v += numerator / denominator
            dissonances.append(diss_v)
        return torch.stack(dissonances, dim=1)

    def Evidence_DC(self, alpha,vacuity):
        E = dict()
        for v in range(len(alpha)):
            E[v] = alpha[v]-1
            E[v] = torch.nan_to_num(E[v], 0)
        for v in range(len(alpha)):
            E[v] = torch.nan_to_num(E[v], 0)
        E_con = E[0]
        for v in range(1, len(alpha)):
            E_con = torch.min(E_con, E[v])
        for v in range(len(alpha)):
            E[v] = torch.sub(E[v], E_con)
            E[v] = torch.mul(E[v], vacuity[:, v].unsqueeze(1))
        alpha_con = E_con + 1
        E_div = E[0]
        for v in range(1,len(alpha)):
            E_div = torch.add(E_div, E[v])
        S_con = torch.sum(alpha_con, dim=1, keepdim=True)
        b_con = torch.div(E_con, S_con)
        S_b = torch.sum(b_con, dim=1, keepdim=True)
        b_con2 = torch.pow(b_con, 1.25)
        S_b2 = torch.sum(b_con2,dim=1, keepdim=True)
        b_cona = torch.mul(b_con2, torch.div(S_b, S_b2))
        E_con = torch.mul(b_cona, S_con)
        E_con = torch.mul(E_con, len(alpha))
        E_a = torch.add(E_con, E_div)
        alpha_a = E_a + 1
        alpha_con = E_con + 1
        alpha_a = torch.nan_to_num(alpha_a, 0)
        alpha_con = torch.nan_to_num(alpha_con, 0)
        alpha_div = torch.nan_to_num(E_div+1, 0)
        S_a = torch.sum(alpha_a, dim=1, keepdim=True)
        return alpha_a, alpha_con, alpha_div,S_a

    def forward(self, X):
        ps = []
        evidences = dict()
        for v in range(self.num_views):
            evidences[v] = self.EvidenceCollectors[v](X[v])
        alpha = dict()
        for v_num in range(len(X)):
            alpha[v_num] = evidences[v_num] + 1
            S = torch.sum(alpha[v_num], dim=1, keepdim=True)
            p = alpha[v_num] / S
            ps.append(p)
        # 计算不和谐度
        dissonances = self.calculate_dissonance(evidences, self.device)
        vacuity = self.calculate_vacuity(alpha)
        alpha_a, alpha_con, alpha_div,S_a = self.Evidence_DC(alpha,vacuity)
        evidence_a = alpha_a - 1
        evidence_con = alpha_con - 1
        evidence_div = alpha_div - 1
        return evidences, evidence_a, evidence_con, evidence_div,dissonances,vacuity,alpha_a,S_a,ps

class EvidenceCollector(nn.Module):
    def __init__(self, dims, num_classes):
        super(EvidenceCollector, self).__init__()
        self.num_layers = len(dims)
        self.net = nn.ModuleList()
        self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))
        self.net.append(nn.Softplus())

    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return h

