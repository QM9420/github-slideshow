import math
import torch
import torch.nn as nn
import numpy as np

class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, count, valid_length=None):
        d = query.shape[-1]         # 单词向量的维度。
        d2 = query.shape[0]         # 目的词的长度。

        scores = torch.mm(query, torch.transpose(key, 1, 0)) / math.sqrt(d)    # transpose()调换数列的行列值的索引值
        scores_sum = torch.sum(scores, dim=0)
        scores_sum = scores_sum/count
        attention_weights = nn.Softmax(dim=0)
        attention_weights = attention_weights(scores_sum)
        attention_x = attention_weights.reshape(-1, 1)
        attention_h = attention_x * value

        return attention_h, attention_weights

if __name__ == '__main__':
    atten = DotProductAttention(dropout=0)

    keys = torch.rand((17, 400), dtype=torch.float)
    values = keys
    atth, weight = atten(torch.ones((1, 400), dtype=torch.float), keys, values, torch.FloatTensor([1, 17]))
    print(atth)
    print(weight)
