import torch.nn as nn
import torch


class RNN(nn.Module):
    def __init__(self,input_dim,embed_dim,hidden_dim,output_dim):
        super().__init__()

        self.embeds = nn.Embedding(input_dim,embed_dim)
        self.rnn = nn.RNN(embed_dim,hidden_dim)
        self.fc = nn.Linear(hidden_dim,output_dim)

    def forward(self,text):
        x = self.embeds(text)
        output,hidden = self.rnn(x)

        return self.fc(hidden.squeeze(0)).view(-1)


class LSTM(nn.Module):
    def __init__(self,input_dim,embed_dim,hidden_dim,output_dim):
        super().__init__()

        self.embeds = nn.Embedding(input_dim,embed_dim)
        self.lstm = nn.LSTM(embed_dim,hidden_dim)
        self.fc = nn.Linear(hidden_dim,output_dim)

    def forward(self,text,text_length):
        x = self.embeds(text)
        packed = torch.nn.utils.rnn.pack_padded_sequence(x,text_length)

        packed_output, (h,c) = self.lstm(packed)

        return self.fc(h.squeeze(0)).view(-1)



class GRU(nn.Module):
    def __init__(self,input_dim,embed_dim,hidden_dim,output_dim):
        super().__init__()

        self.embeds = nn.Embedding(input_dim,embed_dim)
        self.gru = nn.GRU(embed_dim,hidden_dim)
        self.fc = nn.Linear(hidden_dim,output_dim)

    def forward(self,text,text_length):
        x = self.embeds(text)
        packed = torch.nn.utils.rnn.pack_padded_sequence(x,text_length)

        packed_output, (h,c) = self.gru(packed)

        return self.fc(h.squeeze(0)).view(-1)