import torch.nn as nn
import torch


class ActorCriticNet(nn.Module):
    def __init__(self,input_shape,n_actions):
        super().__init__()
        self.conv =nn.Sequential( 
            nn.Conv2d(input_shape[0],32,8,4),
            nn.ReLU(),
            nn.Conv2d(32,64,4,2),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1),
            nn.ReLU(),
            nn.Flatten()
        )

        conv_out = self.conv(torch.zeros(1,*input_shape)).shape[-1]


        self.policy = nn.Sequential(
            nn.Linear(conv_out,512),
            nn.ReLU(),
            nn.Linear(512,n_actions)
        ) 

        self.value = nn.Sequential(
            nn.Linear(conv_out,512),
            nn.ReLU(),
            nn.Linear(512,1)
        )

    def forward(self,x):
        x = x/255
        conv_out = self.conv(x)
        return self.policy(conv_out),self.value(conv_out)
    
if __name__=="__main__":
    input_shape = (3,64,64)
    x = torch.zeros(1,*input_shape).to('cuda')

    model = ActorCriticNet(input_shape,4).to('cuda')

    policy , value = model(x)

    print(policy.shape,value.shape)