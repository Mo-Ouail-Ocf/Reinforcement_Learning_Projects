import torch.nn as nn
import torch

# Deep mind architecture of 2015
# can be improved with same padding mode -> max pooling -> Leaky ReLU -> max pool
class DQNAgent(nn.Module):
    def __init__(self,input_shape,n_actions:int) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0],32,8,4),
            nn.ReLU(),
            nn.Conv2d(32,64,4,2),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1),
            nn.ReLU(),
            nn.Flatten()
        )
        size =self.conv(torch.zeros(1,*input_shape)).size()[-1]
        self.fc = nn.Sequential(
            nn.Linear(size,512),
            nn.ReLU(),
            nn.Linear(512,n_actions)
        )

    def forward(self,x)->torch.Tensor:
        xx=x/255.0
        return self.fc(self.conv(xx))
    

