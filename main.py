from build import ccb
import torch
a = torch.randn(4).cuda()
b = torch.randn(4).cuda()
c = torch.zeros(4).cuda()
ccb.add_tensor(c, a, b)
print(a)
print(b)
print(c)
torch.testing.assert_close(c, a + b)
