import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_Cell(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size,hidden_size)
        self.U = nn.Linear(input_size,hidden_size)
        self.V = nn.Linear(hidden_size,output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,x,h):
        h = F.tanh(self.W(h) + self.U(x))
        o = self.V(h)
        y = self.softmax(o)

        return y,h

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.rnn = RNN_Cell(input_size=self.input_size,hidden_size=self.hidden_size)
        self.fc = nn.Linear(self.hidden_size,self.output_size)

    def init_hidden(self):
        return torch.zeros(self.num_layers,self.batch_size,self.hidden_size)

    def forward(self,x):
        self.batch_size = x.size(0)
        h0 = self.init_hidden()
        output,hidden = self.rnn(x,h0)
        output = self.fc(output[:,-1,:])
        return output


class LSTM_cell(nn.Module):
    def __init__(self,input_size,output_size,hidden_dim,num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_dim
        self.output_size = output_size
        self.num_layers = num_layers

        self.Ix,self.Ih = self.get_weights()
        self.Fx,self.Fh = self.get_weights()
        self.Ox,self.Oh = self.get_weights()
        self.Cx,self.Ch = self.get_weights()
        # from the paper the weights for the gates



    def get_weights(self):
        Wx = nn.Linear(self.input_size,self.hidden_size,bias=True)
        Wh = nn.Linear(self.hidden_size,self.hidden_size,bias=True)

        return (Wx,Wh)

    def forward(self,inputs,H_C=None):
        if H_C is None:
            H = torch.zeros((inputs.shape[1],self.hidden_size))
            C = torch.zeos((inputs.shape[1],self.hidden_size))
        else:
            H,C = H_C

        outputs = []
        for x in inputs:
            I = F.sigmoid(self.Ix(x)+self.Ih(H))
            F = F.sigmoid(self.Fx(x)+self.Ih(H))
            O = F.sigmoid(self.Ox(x)+self.Oh(H))

            C_tilda = F.tanh(self.Cx(x)+self.Ch(H))

            C = F * C + I * C_tilda
            H = O * torch.tanh(C)

            outputs.append(H)

        return outputs,(H,C)

