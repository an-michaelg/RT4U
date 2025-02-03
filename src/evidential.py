# Functions for evidential loss and combination

import torch
import torch.nn.functional as F

# KL divergence between the dirichlet distribution parameterized by alpha and the uniform dirichlet
# computes KLD per sample, expects input size of NxC
def KL(alpha):
    beta = torch.ones(1, alpha.size()[1], device=alpha.device)
    S_alpha = torch.sum(alpha, 1, keepdim=True)
    S_beta = torch.sum(beta, 1, keepdim=True)
    #print(S_alpha)
    #print(S_beta)
    
    log_gamma_alpha = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), 1, keepdim=True)
    log_gamma_beta = torch.lgamma(S_beta) - torch.sum(torch.lgamma(beta), 1, keepdim=True)
    #print(log_gamma_alpha)
    #print(log_gamma_beta)
    
    dg_a = torch.digamma(alpha)
    dg_a0 = torch.digamma(S_alpha)
    #print(dg_a)
    #print(dg_a0)
    
    kl = torch.sum((alpha-beta)*(dg_a-dg_a0) , 1, keepdim=True) + log_gamma_alpha - log_gamma_beta
    return kl

class EvidentialLoss(torch.nn.Module):
    def __init__(self, weight=None) -> None:
        super().__init__()
        self.weight = weight # weighted loss function for addressing label imbalance
    
    def forward(self, net_output, labels, kl_coeff=0.1):
        # assume output and labels are NxC, labels sum to 1 (eg. one-hot encoding)
        # enforce output non-negativity to yield the evidence values
        evidence = F.relu(net_output)
        alpha = evidence + 1
        
        S = torch.sum(alpha, 1, keepdim=True)
        predicted_mass = alpha/S
        
        sq_err = (labels-predicted_mass)**2
        if self.weight is not None:
            weighted_sq_err = sq_err * self.weight
        else:
            weighted_sq_err = sq_err
            
        sq_loss = torch.sum(weighted_sq_err, 1, keepdim=True)

        dirichlet_variance = alpha*(S-alpha)/(S*S*(S+1))
        var = torch.sum(dirichlet_variance, 1, keepdim=True)

        alpha_tilde = evidence*(1-labels) + 1

        kl = KL(alpha_tilde)
        loss = (sq_loss + var) + kl_coeff * kl
        return loss.mean()
        
        
if __name__ == "__main__":
    coeff = 0.1
    EvLoss = EvidentialLoss()
    logits = torch.Tensor([[-1, -1, 5],[-4, 2.1, 2.1]])
    labels = torch.Tensor([[0, 0, 1],[0.0, 0.5, 0.5]])
    loss = EvLoss(logits, labels, coeff)
    print(loss)
    