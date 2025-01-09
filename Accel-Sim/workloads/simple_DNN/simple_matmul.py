# import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# A = torch.randn(1024, 1024, device=device).half()  # Tensor core
# B = torch.randn(1024, 1024, device=device).half()

# with torch.no_grad():
#     C = torch.matmul(A, B)

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A = torch.randn(1024, 1024, device=device, dtype=torch.float32)  # 단정밀도 사용
B = torch.randn(1024, 1024, device=device, dtype=torch.float32)

with torch.no_grad():
    C = torch.matmul(A, B)
