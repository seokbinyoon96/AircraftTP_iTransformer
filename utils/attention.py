class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, hidden, encoder_outputs):

        hidden = hidden.unsqueeze(2)
        attention_energies = torch.bmm(encoder_outputs, hidden).squeeze(2)

        return F.softmax(attention_energies, dim=1)
