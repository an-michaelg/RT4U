# rather simple gradient descent to figure out Platt and Temperature scalings
import numpy as np
import torch
import torch.nn.functional as F

def platt_scaling_fit(logits, y, num_iters=5000, mode="platt"):
    # logits: (N, D) ndarray of logits
    # y: (N,) ndarray of integer labels, or (N, D) ndarray of probabilities that sum to 1
    N, D = logits.shape
    if mode == "platt": # make sure the number of constraints is above the degrees of freedom
        assert N > D
    logits_tensor = torch.Tensor(logits)
    if mode == "platt": # params are 1/A_k
        params = torch.randn(D, requires_grad=True)
    else: # temperature scaling, params is 1/T
        #params = torch.randn(1, requires_grad=True).to(device)
        #params = torch.tensor(1.01, requires_grad=True) #
        params = torch.normal(mean=torch.Tensor([1]), std=torch.Tensor([0.1]))
        params.requires_grad = True
    y_tensor = torch.Tensor(y).long()
    optimizer = torch.optim.SGD([params], lr=1e-2)
    
    for i in range(num_iters):
        optimizer.zero_grad()
        temp = torch.abs(params + 1)
        new_logits = logits_tensor * temp
        loss = F.cross_entropy(new_logits, y_tensor)
        loss.backward()
        optimizer.step()
        #if i % 500 == 0:
        #  print(f"iteration {i}, loss = {loss.item()}, {temp}")
            
    return temp.detach().numpy()
    
if __name__ == "__main__":
    logits = torch.Tensor([[-1, -2, 5],[-1, 2, 2], [-1, -1, 5],[-1, -1, 3]])
    y = [2, 2, 2, 2]
    scaling = platt_scaling_fit(logits, y, mode="temp")
    print(scaling)