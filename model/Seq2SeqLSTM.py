class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target=None, teacher_forcing_ratio=0.3):
        batch_size = source.size(0)
        output_dim = self.decoder.output_dim
        if target is None:
            target_length = 20
            teacher_forcing_ratio = 0.0
        else:
            target_length = target.shape[1]

        outputs = torch.zeros(batch_size, target_length, output_dim).to(self.device)
        attn_weights_list = [] 

        
        encoder_outputs, hidden, cell = self.encoder(source)

        input = source[:, -1, :3]

        for t in range(target_length):
            output, hidden, cell, attn_weights = self.decoder(input, hidden, cell, encoder_outputs)
       
            outputs[:, t, :] = output.squeeze(1)
            attn_weights_list.append(attn_weights) 
            teacher_force = random.random() < teacher_forcing_ratio
            input = target[:, t, :] if teacher_force and target is not None else output.squeeze(1)

        attn_weights_tensor = torch.stack(attn_weights_list, dim=1)
        return outputs
