import torch
import torch.nn as nn
import torch.nn.functional as F
from embed import DataEmbedding
from encoder import Encoder, EncoderLayer, Dialated_Casual_Conv
from attn import ProbAttention, AttentionLayer


class SAFE_Net(nn.Module):
    def __init__(self, enc_in, c_out, seq_len, out_len,
                factor=5, d_model=64, n_heads=8, e_layers=3, d_ff=512, 
                dropout=0.05, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, use_snn = True):
        super(SAFE_Net, self).__init__()

        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.use_snn = use_snn

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        Attn = ProbAttention
        
        # Encoder#
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                Dialated_Casual_Conv(
                    d_model
                ) for l in range(e_layers)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        self.encoder2 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                Dialated_Casual_Conv(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.encoder3 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                Dialated_Casual_Conv(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )


        self.projection = nn.Linear(d_model, c_out, bias=True)

        self.cls_projection = nn.Linear(seq_len*d_model, 8, bias=True)

        self.weight_simple = nn.Linear(d_model, 1, bias=True)

        
        
    def forward(self, x_enc, x_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out2, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        r1 = enc_out - enc_out2
        enc_out3, attns2 = self.encoder2(enc_out2, attn_mask = enc_self_mask)
        w1 = self.weight_simple(enc_out3)
        enc_out3 = w1 * enc_out3
        r2 = enc_out2 - enc_out3

        enc_out4, attns3 = self.encoder3(r2, attn_mask = enc_self_mask)
        w2 = self.weight_simple(enc_out4)
        enc_out4 = w2 * enc_out4
        r3 = r2 - enc_out4

        subject_agnostic_feature = enc_out3 + enc_out4
        dec_out = self.projection(subject_agnostic_feature)

        domain_feature = (r3).contiguous().view(r3.size(0), -1)
        cls = self.cls_projection(domain_feature)

        orthogonal_loss = torch.mean(torch.abs(torch.sum(torch.mean(subject_agnostic_feature, dim=2) * torch.mean(r3, dim=2), dim=1)))

        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], F.log_softmax(cls, dim=1), orthogonal_loss, attns3, cls
        else:
            return dec_out[:,-self.pred_len:,:], F.log_softmax(cls, dim=1), orthogonal_loss # [B, L, D]





# if __name__ == '__main__':

#     model = SAFE_Net(enc_in=5, c_out=3, seq_len=100, out_len=1)
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#     model = model.to(device)
#     x = torch.rand(10,100,5).to(device)
#     y = torch.rand(10,1).to(device)
#     out = model(x)
#     print(out[0].shape)
#     print(model)