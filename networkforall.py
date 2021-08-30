import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
# from torchsummary import summary

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Network(nn.Module):
    def __init__(self, input_size, hidden_in_dim, hidden_out_dim, output_dim, rnn_num_layers, rnn_hidden_size, device, actor=False):
        super(Network, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""
        self.device = device
        self.rnn_num_layers = rnn_num_layers
        self.rnn_hidden_size = rnn_hidden_size
        # Recurrent NN layers (LSTM)
        # self.rnn = nn.RNN(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
        # self.rnn = nn.GRU(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
        self.rnn = nn.LSTM(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
        
        # Linear NN layers
        if actor == True:
            self.fc1 = nn.Linear(rnn_hidden_size+int(hidden_in_dim/2),hidden_in_dim)
            self.fc0 = nn.Linear(input_size - 2 ,int(hidden_in_dim/2))
        else:
            self.fc1 = nn.Linear(rnn_hidden_size+int(hidden_in_dim/2),hidden_in_dim)
            self.fc0 = nn.Linear(input_size,int(hidden_in_dim/2))
        self.fc2 = nn.Linear(hidden_in_dim,hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim,output_dim-1)
        self.fc4 = nn.Linear(hidden_out_dim,1)
        self.nonlin = f.relu #leaky_relu
        self.nonlin_tanh = torch.tanh #tanh
        self.actor = actor
        #self.reset_parameters()

    def reset_parameters(self):
        self.rnn.weight.data.uniform_(*hidden_init(self.rnn))
        self.fc0.weight.data.uniform_(*hidden_init(self.fc0))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)
        self.fc4.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x1, x2):
        if self.actor:
            # return a vector of the force
            # RNN
            h0 = torch.zeros(self.rnn_num_layers, x1.size(0), self.rnn_hidden_size).to(self.device) #Initial values for RNN
            c0 = torch.zeros(self.rnn_num_layers, x1.size(0), self.rnn_hidden_size).to(self.device) #Initial values for RNN
            # out, _ = self.rnn(x1,h0)
            out, _ = self.rnn(x1,(h0,c0))
            # out: batch_size, seq_legnth, hidden_size
            out = out[:,-1,:]
            # out: batch_size, hidden_size
            h00 = self.nonlin(self.fc0(x2))
            x = torch.cat((out,h00), dim=1)
            # Linear
            h1 = self.nonlin(self.fc1(x))
            h2 = self.nonlin(self.fc2(h1))
            h3 = self.nonlin_tanh(self.fc3(h2))
            h4 = self.nonlin(self.fc4(h2))
            
            # h3 is a 2D vector (a force that is applied to the agent)
            # we bound the norm of the vector to be between 0 and 10
            # norm = torch.norm(h3)
            # return 10.0*(torch.tanh(norm))*h3/norm if norm > 0 else 10*h3
            # return 1.0*(torch.tanh(norm))*h3/norm if norm > 0 else 1*h3
            
            #New configuration where we take into acount the angular velocity and forward velocity.
            #h3 is the angular force applied to the agnet, due to the tanh activation layer, its bounded between -1 and 1, we reset these bounds to -10, 10.
            h3 = h3*10.
            #h4 is the forward force applied to the agent, we bound this to 1.
            norm = torch.norm(h4)
            h4 = 1.0*(torch.tanh(norm))*h4/norm if norm > 0 else 1.0*h4
            return torch.cat((h3,h4), dim=1)
        
        else:
            # critic network simply outputs a number
            # RNN
            h0 = torch.zeros(self.rnn_num_layers, x1.size(0), self.rnn_hidden_size).to(self.device) #Initial values for RNN
            c0 = torch.zeros(self.rnn_num_layers, x1.size(0), self.rnn_hidden_size).to(self.device) #Initial values for RNN
            # out, _ = self.rnn(x1,h0)
            out, _ = self.rnn(x1,(h0,c0))
            # out: batch_size, seq_legnth, hidden_size
            out = out[:,-1,:]
            # out: batch_size, hidden_size
            h00 = self.nonlin(self.fc0(x2))
            x = torch.cat((out,h00), dim=1)
            # Linear
            h1 = self.nonlin(self.fc1(x))
            h2 = self.nonlin(self.fc2(h1))
            h3 = self.nonlin_tanh(self.fc3(h2))
            h4 = self.nonlin(self.fc4(h2))
            return torch.cat((h3,h4), dim=1)


# from tensorboardX import SummaryWriter
# logger = SummaryWriter(log_dir='test')
# logger.add_graph(Network)

# num_landmarks = 1
# num_agents = 1
# in_actor = num_landmarks*2 + (num_agents-1)*2 + 2+2 + num_landmarks + 2 #x-y of landmarks + x-y of others + x-y and x-y velocity of current agent + range to landmarks + 2 actions
# hidden_in_actor = in_actor*15
# hidden_out_actor = int(hidden_in_actor/2)
# out_actor = 2 #each agent have 2 continuous actions on x-y plane
# in_critic = in_actor * num_agents # the critic input is all agents concatenated
# hidden_in_critic = in_critic * 15 + out_actor * num_agents
# hidden_out_critic = int(hidden_in_critic/2)
# #RNN
# rnn_num_layers = 2 #two stacked RNN to improve the performance (default = 1)
# rnn_hidden_size_actor = hidden_in_actor
# rnn_hidden_size_critic = hidden_in_critic - out_actor * num_agents
        
# import hiddenlayer as hl
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, rnn_num_layers, rnn_hidden_size_actor, device,actor=True)
# hl.build_graph(model,torch.zeros([1,2,3,4]))


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, rnn_num_layers, rnn_hidden_size_actor, device,actor=True).to(device)
# summary(model, (1, 28, 28))