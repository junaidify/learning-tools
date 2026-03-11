import torch

device = torch.device('cuda')

# scalar = torch.tensor(7, device=device)
# vector = torch.tensor([1, 2, 3])
# matrix = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
# tensor = torch.tensor([[1, 3], [4, 5], [6, 7]])

# print(f"Executing tensor with GPU: {matrix}")
# print(f"Shape of tensors: {matrix.shape}")
# print(f"Number of dimensions: {matrix.ndim}")

# print(f"Change the shape of matrix: {matrix.reshape(3, 2)}")


# Day - 2

size = 1000

x = torch.randn(size, size)
y = torch.randn(size, size)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)


start_event.record()
result = x @ y
end_event.record()

torch.cuda.synchronize()

time = start_event.elapsed_time(end_event)

print(f"Took: {time:.2f} ms")


