class iTransformer(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, n_heads, e_layers, d_ff, dropout, activation, embed, freq, output_attention, use_norm, class_strategy, factor):
        super(ITSTransformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.use_norm = use_norm
        self.d_model = d_model
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout)
        #self.conv_layer = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=7, stride=6)

        # Encoder-only architecture
        self.cor_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )



        self.projector = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, self.pred_len)
        )

        

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = (x_enc - means) / stdev
  

        enc_out = self.enc_embedding(x_enc, x_mark_enc if x_mark_enc is not None and x_mark_enc.size(2) > 0 else None)
 
        enc_out, cor_attn = self.cor_encoder(enc_out)
        dec_out_mean = self.projector(enc_out)
        
        if self.use_norm:
            dec_out_mean = (dec_out_mean.transpose(1, 2) * stdev) + means
   

        return dec_out_mean, cor_attn
