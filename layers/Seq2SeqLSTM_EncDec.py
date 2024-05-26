import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers,dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
 
        return outputs, hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)  
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2) 
        self.fc3 = nn.Linear(hidden_dim//2, output_dim)  


    def forward(self, x, hidden, cell, encoder_outputs):
        last_hidden = hidden[-1]
        attn_weights = self.attention(last_hidden, encoder_outputs)  
   
        attn_weights = attn_weights.unsqueeze(1)

        weighted = torch.bmm(attn_weights, encoder_outputs)
        output, (hidden, cell) = self.lstm(x.unsqueeze(1), (hidden, cell))
        concatenated_output = torch.cat((output.squeeze(1), weighted.squeeze(1)), dim=1) # 
        fc1 = F.relu(self.fc1(concatenated_output))
        fc2 = F.relu(self.fc2(fc1))
        prediction = self.fc3(fc2)
        return prediction, hidden, cell, attn_weights

