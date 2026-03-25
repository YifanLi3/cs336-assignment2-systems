import torch

# Case 1: FP32 + FP32
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(f'FP32 + FP32:          {s}')

# Case 2: FP16 + FP16
s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(f'FP16 + FP16:          {s}')

# Case 3: FP32 + FP16 (auto promote)
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(f'FP32 + FP16 (auto):   {s}')

# Case 4: FP32 + FP16->FP32 (explicit cast)
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(f'FP32 + FP16->FP32:    {s}')

print(f'\nExpected: 10.0')