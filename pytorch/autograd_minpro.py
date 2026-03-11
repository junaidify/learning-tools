import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.tensor(0.0, requires_grad=True, device=device)
learning_rate = 0.1


for step in range(20): 
    y = (x - 4) ** 2

    y.backward()

    print(f"{step:<8} {x.item():<15.4f} {y.item():<15.4f} {x.grad.item():.4f}")

    with torch.no_grad(): 
        x -= learning_rate * x.grad

    x.grad.zero_()

print(f"\nFinal x = {x.item():.4f} (should be close to 4.0)")