import torch
from torch import nn, optim
import torch.nn.functional as F


class PreLayer(nn.Module):
    def __init__(self, num_id,in_size,emb_size,grap_size,dropout):
        super(PreLayer, self).__init__()
        self.emb_size = emb_size
        self.num_id = num_id
        self.emb = nn.Conv1d(in_channels=in_size,out_channels=emb_size,kernel_size=1)
        self.emb2 = nn.Linear(num_id, num_id)
        self.att = FSM(emb_size, emb_size)

        self.linear1 = nn.Conv1d(in_channels=emb_size,out_channels=emb_size,kernel_size=1,bias= False)
        self.linear2 = nn.Conv1d(in_channels=emb_size,out_channels=emb_size,kernel_size=1,bias= True)

        self.layernorm = nn.LayerNorm([emb_size,num_id])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, ct: torch.Tensor,graph_data: torch.Tensor):
        ### embeding and Inductive attention
        x = F.leaky_relu(self.emb(x))
        x = x.permute(0, 2, 1)
        x = self.att(x) + x
        x = x.permute(0, 2, 1)

        # Predefined graph
        graph_data1 = graph_data[0].to(x.device) + torch.eye(self.num_id).to(x.device)
        graph_data1 = PreLayer.calculate_laplacian_with_self_loop(graph_data1)
        graph_data2 = graph_data[1].to(x.device) + torch.eye(self.num_id).to(x.device)
        graph_data2 = PreLayer.calculate_laplacian_with_self_loop(graph_data2)

        x_new =     self.layernorm(self.linear1(self.dropout(self.linear1(x)) @ graph_data1) @ graph_data2)
        ft = F.gelu(self.layernorm(self.linear2(self.dropout(self.linear2(x)) @ graph_data1) @ graph_data2))
        rt = F.gelu(self.layernorm(self.linear2(self.dropout(self.linear2(x)) @ graph_data1) @ graph_data2))
        ct = ft * ct + x_new - ft * x_new
        ht = rt * F.elu(ct) + x - rt * x
        return ht, ct

    @staticmethod
    def calculate_laplacian_with_self_loop(matrix):
        row_sum = matrix.sum(1)
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_laplacian = (
            matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
        )
        return normalized_laplacian

class DyLayer(nn.Module):
    def __init__(self, num_id,in_size,emb_size,grap_size,dropout):
        super(DyLayer, self).__init__()
        self.emb_size = emb_size
        self.num_id = num_id
        self.emb = nn.Conv1d(in_channels=in_size,out_channels=emb_size,kernel_size=1)
        self.emb2 = nn.Linear(num_id, num_id)
        self.att = FSM(emb_size, emb_size)

        self.linear1 = nn.Conv1d(in_channels=emb_size,out_channels=emb_size,kernel_size=1,bias= False)
        self.linear2 = nn.Conv1d(in_channels=emb_size,out_channels=emb_size,kernel_size=1,bias= True)

        self.layernorm = nn.LayerNorm([emb_size,num_id])
        self.dropout = nn.Dropout(dropout)

        self.GL = nn.Parameter(torch.FloatTensor(num_id,grap_size))
        nn.init.kaiming_uniform_(self.GL)
        self.GL_linear = nn.Linear(grap_size, emb_size,bias=False)
        self.GL_linear2 = nn.Linear(emb_size * 2, emb_size * 2 ,bias=False)

    def forward(self, x: torch.Tensor, ct: torch.Tensor,gearph_data: torch.Tensor):

        x = F.leaky_relu(self.emb(x))
        x = x.permute(0, 2, 1)
        x = self.att(x) + x
        x = x.permute(0, 2, 1)

        B, _, _ = x.shape
        GL_embed = self.GL_linear(self.GL.unsqueeze(0).expand(B, -1, -1))
        GL_embed = self.GL_linear2(torch.cat([x.transpose(-2, -1), GL_embed], dim=-1))
        graph_learn = torch.eye(self.num_id).to(x.device) + F.softmax(F.relu(GL_embed @ GL_embed.transpose(-2, -1)),dim=-1)
        graph_learn = torch.eye(self.num_id).to(x.device) + F.softmax(F.relu(self.GL @ self.GL.transpose(-2, -1)),dim=-1)

        x_new =     self.layernorm(self.dropout(self.linear1(x)) @ graph_learn)
        ft = F.gelu(self.layernorm(self.dropout(self.linear2(x)) @ graph_learn))
        rt = F.gelu(self.layernorm(self.dropout(self.linear2(x)) @ graph_learn))

        ct = ft * ct + x_new - ft * x_new
        ht = rt * F.elu(ct) + x - rt * x
        return ht, ct


class FSM(nn.Module):
    def __init__(self, d_c, d_f):
        super(FSM, self).__init__()

        self.fsm1 = nn.Linear(d_c, d_c)
        self.fsm2 = nn.Linear(d_c, d_f)
        self.fsm3 = nn.Linear(d_c + d_f, d_c)
        self.fsm4 = nn.Linear(d_c, d_c)

    def forward(self, input: torch.Tensor):
        b, n, _ = input.shape
        
        com_feat = F.gelu(self.fsm1(input))
        com_feat = self.fsm2(com_feat)

        if self.training:
            # ,b,n,d
            ratio = F.softmax(com_feat, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, n)#b*d,n
            indices = torch.multinomial(ratio, 1)#b*d,1
            indices = indices.view(b, -1, 1).permute(0, 2, 1)#b,1,d
            com_feat = torch.gather(com_feat, 1, indices)#b,1,d
            com_feat = com_feat.repeat(1, n, 1)#b,n,d
        else:
            weight = F.softmax(com_feat, dim=1)
            com_feat = torch.sum(com_feat * weight, dim=1, keepdim=True).repeat(1, n, 1)

        # mlp fusion
        com_feat_new = torch.cat([input, com_feat], -1)
        com_feat_new = F.gelu(self.fsm3(com_feat_new))
        output = self.fsm4(com_feat_new)

        return output
    