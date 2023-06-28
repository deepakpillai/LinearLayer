import torch
from torch import nn

class LinearRegressionModelClass(nn.Module):
  def __init__(self):
    super().__init__()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # self.weight = nn.Parameter(torch.randn(1, device=device, dtype=torch.float32, requires_grad=True))
    # self.bias = nn.Parameter(torch.randn(1, device=device, dtype=torch.float32, requires_grad=True))
    self.linear_layer = nn.Linear(1,1, device=device, dtype=torch.float32)

  def forward(self, data: torch.Tensor)->torch.Tensor:
    # return (self.weight * data) + self.bias
    return self.linear_layer(data)