import torch
import torch.nn as nn
from torch.cuda.amp import autocast

torch.backends.cudnn.enabled = False

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 단일 Linear 계층 모델 정의
class SingleLinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SingleLinearLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# 모델 초기화 및 FP16으로 변환 후 GPU로 이동
input_dim = 1024
output_dim = 1024
model = SingleLinearLayer(input_dim, output_dim).to(device).half()  # FP16으로 Tensor Core 사용

# 예제 입력 데이터 생성 및 FP16 변환 후 GPU로 이동
dummy_data = torch.randn(64, input_dim).to(device).half()  # 배치 크기 64의 입력 텐서 생성

# 추론(인퍼런스) 수행
with torch.no_grad():
    output = model(dummy_data)
    print("Output shape:", output.shape)
