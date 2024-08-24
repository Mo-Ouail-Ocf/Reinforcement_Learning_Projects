import torch.nn as nn
from torchrl.modules import NoisyLinear
import torch
import typing as tt
from torchsummary import summary
# The model combines :

# 1- N-Step deep q learning : make sure to pass gamma**n to the loss

# 2- Dueling DQN (DDQN) : the model will have two paths : state value (1  output) , advantage function (n_actions output) 
#    then finally Q = V + A - A.mean()

# 3- Noisy DQN : Model will contain noisy layers ;  no need to Epsilon-Greedy policy 

# 4- Prioritirized replay buffer : make sure to update the priorities after calculating the loss

class RainbowDQN(nn.Module):
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
        size = self.conv(torch.zeros(1,*input_shape)).size()[-1]

        self.noisy_layers = [
            NoisyLinear(size,256),
            NoisyLinear(256,n_actions)
        ]

        self.fc_advantage = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
        )

        self.fc_value = nn.Sequential(
            nn.Linear(size,256),
            nn.ReLU(),
            nn.Linear(256,1)
        )

    def get_advanage_value(self,x)->tt.Tuple[torch.Tensor,torch.Tensor]:
        x=x/255.0
        conv = self.conv(x)
        advantage = self.fc_advantage(conv)
        value = self.fc_value(conv)
        return advantage,value
    
    def forward(self,x):
        advantage,value = self.get_advanage_value(x)
        mean = advantage.mean(dim=1,keepdim=True)
        return value+advantage-mean
    
    def reset_noise(self):
        for layer in self.noisy_layers:
            layer.reset_noise()

    @torch.no_grad
    def noisy_layers_snr(self)->tt.List[float]:
        return [
            ((layer.weight_mu ** 2).mean().sqrt() /
             (layer.weight_sigma ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]  
        

if __name__=="__main__":
    model = RainbowDQN((4,64,64),4).to('cuda')
    print(summary(model,(4,64,64))) 
