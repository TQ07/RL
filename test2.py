import torch 



a = torch.tensor([[1., 10.,12.], [1., 9., 7.]])

softmax = torch.nn.Softmax(dim=1)

print(softmax(a))