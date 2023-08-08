import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

from .utils import build_mlp

class Classifier_network(nn.Module):

    def __init__(self, state_shape, hidden_units=(100, 100),
     hidden_activation=nn.Tanh()):
        super().__init__()
        self.net = build_mlp(
            input_dim=2*state_shape[0] + 3,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
    
    def forward(self, states,next_states, rewards,costs,log_pi):
        input = torch.cat([states, log_pi,next_states,rewards,costs], dim=-1)
        return self.net(input)

    def get_confident_sigmoid(self, states,next_states, rewards,costs,log_pi):
        input = torch.cat([states, log_pi,next_states,rewards,costs], dim=-1)
        return F.sigmoid(self.net(input))

    def get_confident_tanh(self, states,next_states, rewards,costs,log_pi):
        input = torch.cat([states, log_pi,next_states,rewards,costs], dim=-1)
        return F.tanh(self.net(input))