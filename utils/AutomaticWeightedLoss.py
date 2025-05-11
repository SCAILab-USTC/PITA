# -*- coding: utf-8 -*-
# https://github.com/Mikoto10032/AutomaticWeightedLoss/blob/master/AutomaticWeightedLoss.py

import torch
import torch.nn as nn
import torch.nn.init as init
import math
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        self.epsilon = 1e-6

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2 + self.epsilon) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

class AdvancedAutoWeightedLoss(nn.Module):
    def __init__(self, num_losses=2, init_method='normal', epsilon=1e-8, 
                grad_clip=None, param_constraint='softplus'):
        super().__init__()
        
        self.param_constraint = param_constraint
        self.grad_clip = grad_clip
        self.epsilon = epsilon
        self.params = nn.Parameter(torch.empty(num_losses))
        
        if init_method == 'kaiming':
            with torch.no_grad():
                tmp = self.params.view(-1, 1)
                init.kaiming_uniform_(tmp, a=math.sqrt(5))
                self.params.data = tmp.view(-1)
        elif init_method == 'xavier':
            with torch.no_grad():
                tmp = self.params.view(-1, 1)
                init.xavier_uniform_(tmp)
                self.params.data = tmp.view(-1)
        elif init_method == 'normal':
            init.normal_(self.params, mean=1.0, std=0.1)
        else:
            init.ones_(self.params)
        self.register_buffer('weight_history', torch.zeros(num_losses, 1000))
        
    def _apply_constraint(self, x):
        if self.param_constraint == 'softplus':
            return nn.functional.softplus(x)
        elif self.param_constraint == 'abs':
            return torch.abs(x)
        elif self.param_constraint == 'exp':
            return torch.exp(x)
        else:
            return x

    def forward(self, *losses):
        constrained_params = self._apply_constraint(self.params)
        weights = 1.0 / (constrained_params.pow(2) + self.epsilon)
 
        self.weight_history = torch.roll(self.weight_history, -1, dims=1)
        self.weight_history[:, -1] = weights.detach()
        
        total_loss = 0
        for i, loss in enumerate(losses):
            total_loss += 0.5 * weights[i] * loss 
            total_loss += torch.log1p(constrained_params[i].pow(2))
            
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            
        return total_loss
    
    
if __name__ == '__main__':
    awl = AdvancedAutoWeightedLoss(
        num_losses=3,
        init_method='kaiming',
        grad_clip=1.0,
        param_constraint='softplus'
    )
    
    loss1 = torch.tensor(1.0, requires_grad=True)
    loss2 = torch.tensor(2.0, requires_grad=True)
    loss3 = torch.tensor(0.5, requires_grad=True)
    
    total_loss = awl(loss1, loss2, loss3)
    total_loss.backward()
    
    print("Current weights:", awl.get_weights())
