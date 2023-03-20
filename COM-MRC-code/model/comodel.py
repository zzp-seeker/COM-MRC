import torch
import torch.nn as nn

from transformers import BertModel,BertTokenizer
from model.attention import MultiHeadAttention
from model.layernorm import LayerNorm

class comModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.embed = BertModel.from_pretrained(args.bert_type,cache_dir=args.cache_path)
        self.embed_dropout = nn.Dropout(0.1)
        embed_size = self.embed.config.hidden_size

        self.a_rep = nn.Linear(embed_size,embed_size,bias=False)
        self.a_ffn = nn.Sequential(
            nn.Linear(embed_size,2,bias=False)
        )

        self.o_rep = nn.Linear(embed_size,embed_size,bias=False)
        self.o_ffn = nn.Sequential(
            nn.Linear(embed_size,2,bias=False)
        )

        self.s_att = MultiHeadAttention(2,768,768,768,768)
        self.s_norm = LayerNorm(768)
        self.layer_dropout = nn.Dropout(0.1)
        self.s_ffn = nn.Sequential(
            nn.Linear(embed_size,3)
        )

        self.is_on_ffn = nn.Sequential(
            nn.Linear(3*embed_size,2),
        )

    def get_aspect(self,inputs,args):
        bert_feature = self.embed(**inputs).last_hidden_state # (40,54,768)
        bert_feature = self.embed_dropout(bert_feature)

        sentence_span = bert_feature[:,args.sen_pre_len:,:] # (40,44,768)

        # aspect
        a_rep = self.a_rep(sentence_span)
        a_logits = self.a_ffn(a_rep) # (40,44,2)

        as_p, ae_p = a_logits.split(1, dim=-1)
        # B++ x S, B++ x S
        as_p = as_p.squeeze(-1)
        ae_p = ae_p.squeeze(-1)

        return as_p,ae_p,sentence_span,a_rep,bert_feature


    def forward(self,inputs,as_index,ae_index,args,plus=None):
        if plus==None:
            as_p,ae_p,sentence_span,a_rep,bert_feature = self.get_aspect(inputs,args)
        else:
            as_p,ae_p,sentence_span,a_rep,bert_feature = plus

        # opinion
        o_rep = self.o_rep(sentence_span)
        o_logits = self.o_ffn(o_rep)

        os_p,oe_p = o_logits.split(1, dim=-1)
        os_p = os_p.squeeze(-1)
        oe_p = oe_p.squeeze(-1)

        # sentiment
        s_y = self.s_att(sentence_span,a_rep,o_rep)
        s_x = self.s_norm(self.layer_dropout(sentence_span + s_y))
        s_x = torch.max(s_x,dim=1)[0]
        s_logits = self.s_ffn(s_x)

        # is_on
        a_x_pooling = torch.max(a_rep,dim=1)[0]
        o_x_pooling = torch.max(o_rep,dim=1)[0]
        is_on_logits = self.is_on_ffn(torch.cat((bert_feature[:,0,:],a_x_pooling,o_x_pooling),1))

        return {'as_p':as_p,'ae_p':ae_p,
                'is_on_logits':is_on_logits,'os_p':os_p,'oe_p':oe_p,
                's_logits':s_logits}

