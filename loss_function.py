import torch
import torch.nn.functional as F
import torch.nn as nn

def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def kl_loss(alpha, y, epoch_num, num_classes, annealing_step, device):
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return torch.mean(kl_div)

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device, useKL=False):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
    if not useKL:
        return A
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div

def Penalty(alpha, target,dissonances,device):
    alpha = alpha.to(device)
    target = target.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    inner = torch.sum(target * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    alpha.requires_grad_(True)
    grad = torch.autograd.grad(
        outputs=inner,
        inputs=alpha,
        grad_outputs=torch.ones_like(inner),
        create_graph=True,
        retain_graph=True
    )[0]
    g_weights = (1 - dissonances) / ((1 - dissonances).sum())
    b = torch.exp(1 - g_weights)
    penalty = torch.mean(grad ** 2 * b.unsqueeze(1))
    return penalty

def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)
    return torch.mean(loss)

def vacuity_loss(alpha, K, epsilon=1e-8):
    S = torch.sum(alpha, dim=1)
    loss_terms = torch.log(K / (S + epsilon) + epsilon)
    return torch.mean(loss_terms)

def class_contrastive_loss(evidence_con, evidence_div, labels, temperature=1):
    batch_size = evidence_con.shape[0]
    labels = labels.view(-1)
    sim_matrix = F.cosine_similarity(
        evidence_con.unsqueeze(1),
        evidence_div.unsqueeze(0),
        dim=2
    ) / temperature
    label_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    pos_samples = -torch.log(torch.sigmoid(sim_matrix)) * label_mask
    neg_samples = -torch.log(1 - torch.sigmoid(sim_matrix)) * (1 - label_mask)
    loss = (pos_samples.sum() + neg_samples.sum()) / (batch_size ** 2)
    return loss

def get_loss(dissonances,evidences, evidence_con, evidence_div, target, epoch_num, num_classes, annealing_step, vartheta, zeta,device):
    cls_loss = class_contrastive_loss(evidence_con,evidence_div,target)
    target = F.one_hot(target, num_classes)
    alpha_con = evidence_con + 1
    alpha_div = evidence_div + 1
    loss_acc = 0
    for v in range(len(evidences)):
        alpha = evidences[v] + 1
        loss_acc += edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device)
        loss_acc += Penalty(alpha,target,dissonances[:,v],device)
        loss_acc += kl_loss(alpha, target, epoch_num, num_classes, annealing_step, device)
    loss_acc += edl_digamma_loss(alpha_con, target, epoch_num, num_classes, annealing_step, device)
    loss_acc += edl_digamma_loss(alpha_div, target, epoch_num, num_classes, annealing_step, device)
    loss_acc += kl_loss(alpha_con, target, epoch_num, num_classes, annealing_step, device)
    loss_acc += vartheta * vacuity_loss(alpha_con, K=num_classes) + zeta * cls_loss
    loss = loss_acc
    if loss.isnan():
        print(loss)
    return loss

def mask_correlated_samples(batch_size):
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=bool)
    mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask

def forward_label(q_i, q_j,dissonance_i,dissonance_j,device):
    temperature_l = 1.0
    lambda_diss = 0.5
    batch_size = q_i.shape[0]
    q = torch.cat([q_i, q_j], dim=0)
    diss_i_full = torch.cat([dissonance_i, dissonance_i], dim=0)
    diss_j_full = torch.cat([dissonance_j, dissonance_j], dim=0)
    # 计算样本间不和谐度权重矩阵
    diss_weights = 1 + lambda_diss * (diss_i_full.unsqueeze(1) + diss_j_full.unsqueeze(0))
    # 计算余弦相似度矩阵
    sim = F.cosine_similarity(q.unsqueeze(1), q.unsqueeze(0), dim=2)
    sim_i_j = sim[:batch_size, batch_size:].diag()
    sim_j_i = sim[batch_size:, :batch_size].diag()
    positive_clusters = torch.cat([sim_i_j, sim_j_i], dim=0)
    # 应用不和谐度权重到负样本
    neg_mask = mask_correlated_samples(batch_size)
    weighted_sim = sim * diss_weights
    neg_sim = weighted_sim[neg_mask].view(2 * batch_size, -1)
    logits = torch.cat([positive_clusters.unsqueeze(1), neg_sim], dim=1) / temperature_l
    labels = torch.zeros(2 * batch_size).to(device).long()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss = criterion(logits, labels)
    loss /= (2 * batch_size)
    return loss


