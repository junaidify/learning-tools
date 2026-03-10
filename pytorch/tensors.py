import torch

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# x = torch.tensor([1.0, 2.0, 3.0]) # first created on CPU then
# x = x.to(device)                  # moved to GPU using to(device) method

# print(f"Tensor: {x}")



# directly creating at GPU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.tensor([1.0, 2.0, 3.0], device=device)
print(f"Tensor is on: {x.device}")