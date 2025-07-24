import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False): # False确保输入张量为(seq_len, batch_size, hidden_size)
        # 确保Attention类继承父类nn.Module所有功能。类的初始化
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        # apply attention layer，bmm计算点积
        weights = torch.bmm(inputs, # 注意力权重完成下列操作后与input进行点乘
                            self.att_weights  # (1, hidden_size)，可学习的注意力权重向量
                            .permute(1, 0)  # (hidden_size, 1)，转置
                            .unsqueeze(0)  # (1, hidden_size, 1)，添加一个维度
                            .repeat(batch_size, 1, 1)  # (batch_size, hidden_size, 1)，将权重复制batch_size次
                            )

        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1) # 计算注意力分数

        # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(), requires_grad=True).cuda() # 创建全为1的张量
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len: # 遍历实际长度l，如果小于最大长度max_len则将mask中对应第i个输入的l之后的值设为0
                mask[i, l:] = 0

        # apply mask and renormalize attention scores (weights) 
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1) # sums per row

        attentions = masked.div(_sums) # 重新归一化attention scores

        # if attentions.dim() == 1:
        #     attentions = attentions.unsqueeze(1)

        # apply attention weights，attentions与inputs逐元素相乘
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze() # 对加权后的输入在第1维（时间步）上求和，得到输出长度固定的特征向量，同时移除多余维度

        return representations, attentions


class MyLSTM(nn.Module): # 结合注意力机制的LSTM
    def __init__(self, embedding_dim=768, hidden_dim=128, lstm_layer=1, dropout=0.6):
        super(MyLSTM, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim
        self.lstm1 = nn.LSTM(input_size=self.embedding_dim,
                             hidden_size=hidden_dim,
                             num_layers=lstm_layer,
                             bidirectional=True)
        self.atten1 = Attention(hidden_dim*2, batch_first=True)  # 2 is bidrectional

    def forward(self, x, x_len):
        x = self.dropout(x) # 丢弃部分特征，防止过拟合

        x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False) # 压缩输入序列
        out, (h_n, c_n) = self.lstm1(x)
        x, lengths = nn.utils.rnn.pad_packed_sequence(out, batch_first=True) # 解压缩输出序列
        x, _ = self.atten1(x, lengths)  # skip connect

        return x, _


class RedditModel(nn.Module): # 结合LSTM和全连接层
    def __init__(self, op_units=5, embedding_dim=768, hidden_dim=128, lstm_layer=1, dropout=0.5):
        super(RedditModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)

        self.fc_1 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.fc_2 = nn.Linear(hidden_dim*2, op_units)

        self.historic_model = MyLSTM(self.embedding_dim, self.hidden_dim, lstm_layer, dropout)

    def get_pred(self, feat):
        feat = self.fc_1(self.dropout(feat))
        return self.fc_2(feat)

    def forward(self, tweets, lengths, labels):
        h, _ = self.historic_model(tweets, lengths)
        # 正常来说h形状为(batch_size, hidden_dim*2)，h.dim()=2
        # 但如果batch_size=1，会导致h形状为(hidden_dim*2)，h.dim()=1
        if h.dim() == 1:
            h = h.unsqueeze(0) # 通过unsqueeze恢复原来维度，便于后续操作 
        e = self.get_pred(h)

        return e