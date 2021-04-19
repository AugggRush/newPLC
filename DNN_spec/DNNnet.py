import torch
from torch.autograd import Variable
import torch.nn.functional as F


class Dnn_net_Loss(torch.nn.Module):
    def __init__(self):
        super(Dnn_net_Loss, self).__init__()
        
    def forward(self, model_output, targ_input):
        criterion = torch.nn.MSELoss(reduction='mean')
        criterion.cuda()
        targ_input = targ_input.contiguous().view(targ_input.size(0), -1)

        loss = criterion(model_output, targ_input)

        return loss


class DNNnet(torch.nn.Module):
    def __init__(self, n_layer, n_in_channel, n_out_channel):
        super(DNNnet, self).__init__()
        self.n_layer = n_layer
        self.fc_layers = torch.nn.ModuleList()
        self.act_func = torch.nn.LogSigmoid()

        start_layer = torch.nn.Linear(n_in_channel, 2048)
        start_layer = torch.nn.utils.weight_norm(start_layer, name='weight') 
        self.start = start_layer

        for i in range(self.n_layer-2):
            fc_layer = torch.nn.Linear(2048, 2048)
            fc_layer = torch.nn.utils.weight_norm(fc_layer, name='weight')
            self.fc_layers.append(fc_layer)
        
        end_layer = torch.nn.Linear(2048, n_out_channel)
        end_layer = torch.nn.utils.weight_norm(end_layer, name='weight')
        self.end = end_layer
    
    def forward(self, forward_input):
        """
        forward_input = mel spectrongram of 11 input frames: [batchsize , n_mel_channels , frames]
        """
        forward_input = forward_input.contiguous().view(forward_input.size(0), -1)
        output = self.start(forward_input)
        output = self.act_func(output)
        for i in range(self.n_layer-2):
            output = self.fc_layers[i](output)
            output = self.act_func(output)
        output = self.end(output)
        output = self.act_func(output)

        return output
