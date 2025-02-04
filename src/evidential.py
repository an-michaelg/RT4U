# Functions for evidential loss and combination
import torch
import torch.nn.functional as F

# DS combination function from https://github.com/hanmenghan/TMC
def DS_Combin(alpha):
    """
    :param alpha: list of all Dirichlet distribution parameters.
    :return: Combined Dirichlet distribution parameters.
    """
    def DS_Combin_two(alpha1, alpha2):
        """
        :param alpha1: Dirichlet distribution parameters of view 1, NxC
        :param alpha2: Dirichlet distribution parameters of view 2, NxC
        :return: Combined Dirichlet distribution parameters
        """
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        _, num_classes = alpha1.shape
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v]-1
            b[v] = E[v]/(S[v].expand(E[v].shape))
            u[v] = num_classes/S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, num_classes, 1), b[1].view(-1, 1, num_classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
        C = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

        # calculate new S
        S_a = num_classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    for v in range(len(alpha)-1):
        if v==0:
            alpha_a = DS_Combin_two(alpha[0], alpha[1])
        else:
            alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
    return alpha_a

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
        

def discounting_fit(evidence, y, num_iters=5000):
    # evidence: (N, D) ndarray of evidence values
    # y: (N,) ndarray of integer labels, or (N, D) ndarray of probabilities that sum to 1
    N, D = evidence.shape
    
    loss_fcn = EvidentialLoss()
    
    ev_tensor = torch.Tensor(evidence)
    params = torch.randn(1, requires_grad=True)
    
    y_tensor = F.one_hot(torch.Tensor(y).long(), num_classes=D)
    #print(y_tensor)
    optimizer = torch.optim.SGD([params], lr=1e-1)
    
    for i in range(num_iters):
        optimizer.zero_grad()
        discount_factor = F.sigmoid(params)
        new_evidence = discount_factor * ev_tensor
        loss = loss_fcn(new_evidence, y_tensor, kl_coeff=0)
        loss.backward()
        optimizer.step()
        
        #if i % 500 == 0:
        #    print(f"{discount_factor.item()}, {loss.item()}")
            
    return discount_factor.detach().numpy()
    
        
if __name__ == "__main__":
    #coeff = 0.1
    #EvLoss = EvidentialLoss()
    #logits = torch.Tensor([[-1, -1, 5],[-4, 2.1, 2.1]])
    #labels = torch.Tensor([[0, 0, 1],[0.0, 0.5, 0.5]])
    #loss = EvLoss(logits, labels, coeff)
    #print(loss)
    alpha1 = torch.Tensor([[1, 2, 5],[1, 2, 2], [1, 1, 5],[1, 1, 3]])
    alpha2 = torch.Tensor([[1, 1, 5],[1, 1, 3]])
    labels = [2, 0, 0, 2]
    print(discounting_fit(alpha1-1, labels))
    