import torch

model = torch.nn.Linear(10, 10)
for name, param in model.named_parameters():
    if param.requires_grad:
        norm = torch.norm(param)