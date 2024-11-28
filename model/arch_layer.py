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



    
