# rather simple gradient descent to figure out Platt and Temperature scalings
import numpy as np
import torch
import torch.nn.functional as F

def platt_scaling_fit(logits, y, num_iters=5000, mode="platt"):
    # logits: (N, D) ndarray of logits
    # y: (N,) ndarray of integer labels, or (N, D) ndarray of probabilities that sum to 1
    N, D = logits.shape
    assert N > D
    logits_tensor = torch.Tensor(logits)
    if mode == "platt": # params are 1/A_k
        params = torch.randn(D, requires_grad=True)
    else: # temperature scaling, params is 1/T
        params = torch.randn(1, requires_grad=True)
        #params = torch.tensor(1.01, requires_grad=True) #torch.normal(mean=1, std=0, requires_grad=True)
    y_tensor = torch.Tensor(y).long()
    optimizer = torch.optim.SGD([params], lr=1e-3)
    
    for i in range(num_iters):
        optimizer.zero_grad()
        new_logits = logits_tensor * torch.abs(params + 1)
        loss = F.cross_entropy(new_logits, y_tensor)
        loss.backward()
        optimizer.step()
        #if i % 500 == 0:
        #    print(f"iteration {i}, loss = {loss.item()}")
            
    return torch.abs(params + 1).detach().numpy()
    
if __name__ == "__main__":
    D = 10
    N = 500
    volatility = 10
    logits = np.random.randn(N, D) * volatility
    y = np.random.choice(D, N)
    scaling = platt_scaling_fit(logits, y, mode="platt")
    print(scaling)