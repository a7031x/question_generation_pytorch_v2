import torch
import torch.nn as nn

m = nn.LogSoftmax()
loss = nn.NLLLoss(size_average=False, reduce=False)
# input is of size N x C = 3 x 5
input = torch.randn(3, 5, 4, requires_grad=True)
vs = []
for s in torch.unbind(input, 1):
    vs.append(s)
input = torch.cat(vs, 1).view(3, 5, 4)
# each element in target has to have 0 <= value < C
target = torch.tensor([[1,0,0,1],[0,0,1,0],[2,0,1,3]])
output = loss(m(input), target)
output.sum().backward()

print('w', output)