from torch import nn
import torch

class Policy(nn.Module):
    '''
    neural network that gives an action given a state
    '''
    def __init__(self):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(in_features=3, out_features=64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        self.head = nn.Softmax()

    def forward(self, x):
        x = torch.tensor(x)
        x = x.to(torch.float32)
        logits = self.MLP(x)
        return self.head(logits)

class Value(nn.Module):
    '''
    neural network that estimates the expected return/reward at a given state
    '''
    def __init__(self):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(in_features=3, out_features=32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = torch.tensor(x)
        x = x.to(torch.float32)
        logits = self.MLP(x)
        return logits
    
'''policy = Policy()
input = (0,0,0)
output = max(policy(input))

print(output)

output.retain_grad()
output.backward()

params = nn.utils.parameters_to_vector(policy.parameters())
grad = nn.utils.parameters_to_vector(p.grad for p in policy.parameters())


print(grad)'''
value = Value()
input = [(0,0,0),(0,0,0),(0,0,0)]
output = value(input)
print(output)