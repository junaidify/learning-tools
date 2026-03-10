import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

scalar = torch.tensor(7, device=device)
vector = torch.tensor([1,2,3])
matrix = torch.tensor([[1,2,3], [4,5, 6]], device=device)
tensor = torch.tensor([[1,3], [4,5], [6,7]])

print(f"Executing tensor with GPU: {vector}")