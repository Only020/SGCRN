import torch
from torch import nn, optim
import torch.nn.functional as F

from .arch_layer import PreLayer,DyLayer


class SGCRN(nn.Module):
    def __init__(self, input_len, num_id, out_len, 
                 in_size, emb_size,grap_size, layer_num,dropout,adj_mx,
                 if_spatial=16, if_time_in_day=16, if_day_in_week=16
                 ):
        super(SGCRN, self).__init__()

        ### basic parameter
        self.input_len = input_len
        self.out_len = out_len
        self.num_id = num_id
        self.layer_num = layer_num
        self.emb_size = emb_size
        self.graph_data = adj_mx
        # emb
        self.if_spatial = if_spatial
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week

        self.model_size = emb_size
        # spatial embeddings
        if self.if_spatial>0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_id, self.if_spatial))
            nn.init.xavier_uniform_(self.node_emb)
            self.model_size += if_spatial
        # temporal embeddings
        if self.if_time_in_day>0:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(288, self.if_time_in_day))
            nn.init.xavier_uniform_(self.time_in_day_emb)
            self.model_size += if_time_in_day
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(7, self.if_day_in_week))
            nn.init.xavier_uniform_(self.day_in_week_emb)
            self.model_size += if_day_in_week

        self.input_projection = nn.Linear(in_size, emb_size)

        ### encoder
        self.pre_layer = PreLayer(num_id,self.model_size,self.model_size,grap_size,dropout)
        self.dy_layer = DyLayer(num_id, self.model_size, self.model_size, grap_size, dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.lay_norm = nn.LayerNorm([input_len,num_id])

        ### decoder
        self.decoder = nn.Conv2d(in_channels=layer_num,out_channels=out_len,kernel_size=(1,self.model_size))
        self.output = nn.Conv2d(in_channels=out_len,out_channels=out_len,kernel_size=1)


    def forward(self, history_data):
        # Input [B,H,N,C]: B is batch size. N is the number of variables. H is the history length. C is the number of feature.
        # Output [B,L,N]: B is batch size. N is the number of variables. L is the future length

        B, L, N, C = history_data.shape
        
        graph_data = self.graph_data
        tem_emb = []
        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            # In the datasets used in STID, the time_of_day feature is normalized to [0, 1]. We multiply it by 288 to get the index.
            # If you use other datasets, you may need to change this line.
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data* 288).type(torch.LongTensor)]
            tem_emb.append(time_in_day_emb)#B,T,N,D
        else:
            time_in_day_emb = None
            
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data* 7).type(torch.LongTensor)]
            tem_emb.append(day_in_week_emb)#B,T,N,D
        else:
            day_in_week_emb = None #B,emb_size,N
        
        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(B, L, -1, -1))#B,T,N,D
        
        input_emb = self.input_projection(history_data)#B,T,N,D

        hidd = torch.cat([input_emb]+node_emb + tem_emb, dim=-1).permute(0,3,1,2)#B,D,T,N

        ### encoder
        final_result = 0.0
        for z in range(self.layer_num):
            result = 0.0
            ct = torch.zeros(B, self.model_size, N).to(hidd.device)
            if z == 0:
                for j in range(self.input_len):
                    ht, ct = self.pre_layer(hidd[:,:,j,:], ct, graph_data)
                    if j == 0:
                        result = ht.unsqueeze(-2)
                    else:
                        result = torch.cat([result, ht.unsqueeze(-2)], dim=-2)
            else:
                for j in range(self.input_len):
                    ht, ct = self.dy_layer(hidd[:,:,j,:], ct, graph_data)
                    if j == 0:
                        result = ht.unsqueeze(-2)
                    else:
                        result = torch.cat([result, ht.unsqueeze(-2)], dim=-2)

            x = result.clone()
            result = result[:,:,-1,:]
            if z == 0:
                final_result = result.transpose(-2, -1).unsqueeze(1)
            else:
                final_result = torch.cat([final_result, result.transpose(-2, -1).unsqueeze(1)], dim=1)

        ### decoder
        x = self.dropout(self.decoder(final_result))
        x = self.output(x)
        return x.squeeze(-1)

