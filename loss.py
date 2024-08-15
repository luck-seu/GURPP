import torch
from torch.nn import functional as F

def attr_contrastive_loss_op(emb, pos_emb, temperature=0.1):
    temperature = torch.tensor(temperature).to(emb.device)
    batch_size = emb.size(0)
    emb = F.normalize(emb, dim=1)
    pos_emb = F.normalize(pos_emb, dim=1)
    rep = torch.cat([emb, pos_emb], dim=0)
    sim_matrix = F.cosine_similarity(rep.unsqueeze(1), rep.unsqueeze(0), dim=2)
    sim_self = torch.diag(sim_matrix, batch_size)
    sim_pos = torch.diag(sim_matrix, -batch_size)
    positive_samples = torch.cat([sim_self, sim_pos], dim=0)
    nominator = torch.exp(positive_samples / temperature)
    negative_mask = ~torch.eye(2 * batch_size, 2 * batch_size, dtype=bool).to(emb.device)
    denominator = negative_mask * torch.exp(sim_matrix / temperature)
    loss = -torch.log(nominator / torch.sum(denominator, dim=1))
    loss = torch.sum(loss) / (2 * batch_size)
    return loss

def mobility_loss(s_emb, d_emb, mob):
    inner_prod = torch.mm(s_emb, d_emb.T)
    ps_hat = F.softmax(inner_prod, dim=-1)
    inner_prod = torch.mm(d_emb, s_emb.T)
    pd_hat = F.softmax(inner_prod, dim=-1)
    loss = torch.sum(-torch.mul(mob, torch.log(ps_hat)) - torch.mul(mob, torch.log(pd_hat)))
    return loss