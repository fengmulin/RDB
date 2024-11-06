from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import math
BatchNorm2d = nn.BatchNorm2d
class MultiHeadAttention_v2(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention_v2, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        b,c,h,w = query.shape
        query = query.flatten(2).permute(0, 2, 1)
        key = key.flatten(2).permute(0, 2, 1)
        value = value.flatten(2).permute(0, 2, 1)
        # Linear projections
        
        query = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # print(query.shape)
        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = F.softmax(scores, dim=-1)

        # Apply attention to values
        context = torch.matmul(attention, value)
        # print(context.shape)
        # raise
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # Final linear projection
        output = self.out(context).transpose(1, 2).contiguous().view(batch_size,  self.embed_dim, h ,w)
        return output
class MultiHeadAttentionLayer_Fast(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()

        assert hidden_dim % n_heads == 0
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.att = nn.MultiheadAttention(hidden_dim, n_heads, dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.layer_norm(q)
        # print(q.shape, k.shape, v.shape)
        # k = k.view(k.shape[0], k.shape[1], -1)
        # v = v.view(v.shape[0], v.shape[1], -1)
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        # print(q.shape, k.shape, v.shape)
        # raise
        out, att = self.att(q, k, v)

        out = out.reshape(batch_size, -1)

        return out, att
class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim] 
        key --- [N, T_k, key_dim]
        mask --- [N, T_k]
    output:
        out --- [N, T_q, num_units]
        scores -- [h, N, T_q, T_k]
    '''
    def __init__(self, query_dim, key_dim, mid_channel,out_channel, num_heads):
        super().__init__()
        self.num_units = mid_channel
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.W_query = nn.Conv2d(query_dim, self.num_units,kernel_size=1, padding=0)
        self.W_key = nn.Conv2d(key_dim, self.num_units,kernel_size=1, padding=0)
        self.W_value = nn.Conv2d(key_dim, self.num_units,kernel_size=1,padding=0)
        self.out_conv = nn.Conv2d(self.num_units, out_channel,kernel_size=1, stride=1)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(25600)
        self.W_query.apply(self.weights_init)
        self.W_key.apply(self.weights_init)
        self.W_value.apply(self.weights_init)
        self.out_conv.apply(self.weights_init)
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        
    def forward(self, query, key, mask=None):
        # x = query
        batch_size = query.shape[0]
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)
        b, c, h, w = values.shape
        # print(values.shape)
        querys = querys.flatten(2)
        keys = keys.flatten(2)
        values = values.flatten(2)
        # print(querys.shape)
        self.head_dim = self.num_units // self.num_heads
        
        querys = querys.view(8, self.num_heads, self.head_dim, -1)
        keys = keys.view(8, self.num_heads, self.head_dim, -1)
        values = values.view(8, self.num_heads, self.head_dim, -1)
        # querys = torch.stack(torch.split(querys, split_size, dim=1), dim=0)  # [h, N, T_q, num_units/h]
        # keys = torch.stack(torch.split(keys, split_size, dim=1), dim=0)  # [h, N, T_k, num_units/h]
        # values = torch.stack(torch.split(values, split_size, dim=1), dim=0)  # [h, N, T_k, num_units/h]
        ## score = softmax(QK^T / (d_k ** 0.5))
        # print(querys.shape)
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        ## mask
        # print(scores.shape)
        # raise
        
        if mask is not None:
            ## mask:  [N, T_k] --> [h, N, T_q, T_k]
            mask = mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads,1,querys.shape[2],1)
            scores = scores.masked_fill(mask, -np.inf)
        scores = F.softmax(scores, dim=3)
        scores = self.dropout(scores)
        out = torch.matmul(scores, values)
        # print(out.shape)
        # raise
        out = out.view(batch_size, c, h*w)
        out =self.norm(out)
        # out = torch.cat(torch.split(out, 1, dim=0), dim=2).squeeze(0) # [N, T_q, num_units]
        # out = out.permute(0, 2, 1, 3).contiguous()
        # print(out.shape)
        # raise
        out = out.view(b, c,h,w)
        out = self.out_conv(out) 
        # out = self.norm(out+query)
        # print(scores.shape, out.shape)
        # out = scores*values
        return out
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()

        assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)

        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # q = self.layer_norm(q)
        # print(q.shape, k.shape, v.shape)
        q = self.fc_q(q)
        k = self.fc_k(k)
        # v = self.fc_v(v)

        q = q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1,
                                                                        3)
        print(q.shape, 11)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1,
                                                                        3)
        print(k.shape, 22)
        # v = v.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1,
        #                                                                 3)

        att = torch.matmul(q / self.scale, k.permute(0, 1, 3, 2))
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e10)
        att = torch.softmax(att, dim=-1)

        # out = torch.matmul(self.dropout(att), v)
        # out = out.permute(0, 2, 1, 3).contiguous()
        # out = out.view(batch_size, self.hidden_dim)

        # out = self.dropout(self.fc_o(out))

        # return out, att 
        print(att.shape)
        return att.permute(0, 2, 1, 3) 
    
class SAttentionLayer(nn.Module):
    def __init__(self, hidden_dim,  dropout=0.1):
        super().__init__()

        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_key = nn.Conv2d(hidden_dim, hidden_dim,kernel_size=3, padding=1)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.scale = math.sqrt(self.hidden_dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        b, c, h, w = k.shape
        q = q.flatten(2).permute(0, 2, 1)
        k = self.W_key(k)
        k = k.flatten(2)
        v = v.flatten(2).permute(0, 2, 1)
        # q = self.layer_norm(q)
        # print(q.shape, k.shape, v.shape)
        q = self.fc_q(q)
        # k = self.fc_k(k)
        v = self.fc_v(v)

        # q = q.view(batch_size, -1, self.hidden_dim)
        # # print(q.shape, 11)
        # k = k.view(batch_size, -1, self.hidden_dim).permute(0, 2, 1)
        # print(k.shape, 22)
        # v = v.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1,
        #                                                                 3)

        att = torch.matmul(q / self.scale, k)
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e10)
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, v)
        # out = torch.matmul(self.dropout(att), v)
        # out = out.permute(0, 2, 1, 3).contiguous()
        # out = out.view(batch_size, self.hidden_dim)

        # out = self.dropout(self.fc_o(out))

        # return out, att 
        # print(att.shape)
        return out.view(b,c,1,1)
        return att.view(b,1,h,w)
    
class CAttentionLayer(nn.Module):
    def __init__(self, hidden_dim,  dropout=0.1):
        super().__init__()

        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)

        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.scale = math.sqrt(self.hidden_dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        b, c, h, w = k.shape
        q = q.flatten(2)
        k = k.flatten(2).permute(0, 2, 1)
        # q = self.layer_norm(q)
        # print(q.shape, k.shape, v.shape)
        # q = self.fc_q(q)
        k = self.fc_k(k)
        # v = self.fc_v(v)

        q = q.view(batch_size, 1, -1)
        # print(q.shape, 11)
        k = k.view(batch_size, -1, self.hidden_dim)
        # print(k.shape, 22)                                                           

        att = torch.matmul(q / self.scale, k)
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e10)
        att = torch.softmax(att, dim=-1)


        # return out, att 
        # print(att.shape)
        return att.view(b,32,1,1)