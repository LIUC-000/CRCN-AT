import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE hyper-parameters we use in VAT
# n_power: a number of power iteration for approximation of r_vadv
# XI: a small float for the approx. of the finite difference method
# epsilon: the value for how much deviate from original data point X


class AT_X(nn.Module):
    """
    We define a function of regularization, specifically VAT.
    """

    def __init__(self, encoder, classifier):
        super(AT_X, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.n_power = 1
        self.XI = 0.01
        self.epsilon = 1.0

    def forward(self, X, Y):
        x_adv = X + generate_virtual_adversarial_perturbation(X, Y, self.encoder, self.classifier, self.n_power, self.XI, self.epsilon)
        x_adv = x_adv.detach()
        return x_adv  # already averaged

# KL散度
def kl_divergence_with_logit(q_logit, p_logit):
    q = F.softmax(q_logit, dim=1)
    qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
    qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
    return qlogq - qlogp

# 二范数标准化
def get_normalized_vector(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

# 生成对抗干扰r_vadv
def generate_virtual_adversarial_perturbation(x, y, encoder, classifier, n_power, XI, epsilon):
    d = torch.randn_like(x)
    loss = nn.NLLLoss()

    for _ in range(n_power):
        d = XI * get_normalized_vector(d).requires_grad_()
        logit = classifier(encoder(x + d))
        out = F.log_softmax(logit, dim=1)
        loss_adv = loss(out, y)
        grad = torch.autograd.grad(loss_adv, [d])[0]
        d = grad.detach()

    return epsilon * get_normalized_vector(d)