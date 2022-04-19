import pandas as pd
import numpy as np
from LSTM import *
import  torch
from torch.utils.data import TensorDataset,DataLoader,Dataset
from torch import nn
from sklearn.model_selection import train_test_split
import math
class get_DataSet():
    def __init__(self,path,INPUTESIZE=46):
        self.path = path
        self.inputsize = INPUTESIZE
      # LSTM_DATA/1.xlsx
    def dataset_load(self):
        df = pd.read_excel(self.path)
        data = df.to_numpy()
        height, width = data.shape
        # print(height,width) #47 570
        x = data[:-1,:]
        y = data[-1,:]
        # print(x.shape) #(46, 570)
        # print(y.shape)  #(570,)
        x = x.transpose()
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,shuffle=True)
        # print(x_train.shape)  #(399, 46)
        # print(x_test.shape)  #(171, 46)
        # y_train = y_train.reshape(-1,1)
        # y_test = y_test.reshape(-1,1)
        # train_dataset = np.hstack((x_train,y_train))
        # test_dataset = np.hstack((x_test,y_test))
        # print(test_dataset.shape)   #(171, 47)
        # print(train_dataset.shape)   #(399, 47)
        # print(x_train.shape,y_train.shape)   #(399, 46) (399, 1)
        ''' 将输入数据转换为(NUMSAMPLES,TIMESTEP,INPUTSIZE)   '''
        x_train = x_train.reshape(-1,1,self.inputsize)
        y_train = y_train.reshape(-1,1)
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)

        x_test = x_test.reshape(-1,1,self.inputsize)
        y_test = y_test.reshape(-1,1)
        x_test = torch.from_numpy(x_test)
        y_test = torch.from_numpy(y_test)

        train_dataset = TensorDataset(x_train,y_train)
        test_dataset = TensorDataset(x_test,y_test)
        train_loader = DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)
        test_loader = DataLoader(dataset=test_dataset,batch_size=16,shuffle=True)
        # print(1)
        return train_loader,test_loader

class LSTM_modle():
    def __init__(self, input_size, hidden_size, num_layer):
        super().__init__()
        # x [batch, seq_len, hidden_dim * 2]
        # query : [batch, seq_len, hidden_dim * 2]
        # 软注意力机制 (key=value=x)
        self.lstm = nn.LSTM(
            input_size=input_size,  # 特征数量
            hidden_size=hidden_size,  # LSTM hidden unit
            num_layers=num_layer,  # 有几层 LSTM layers
            dropout=0.5

        )
        self.sigmoid = torch.nn.Sigmoid()
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(0.5)
    def attention_net(self,x, query,):
        d_k = query.size(-1)  # d_k为query的维度
        # query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
        # 打分机制 scores: [batch, seq_len, seq_len]
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        # 对最后一个维度 归一化得分
        alpha_n = F.softmax(scores, dim=-1)
        context = torch.matmul(alpha_n, x).sum(1)

        return context, alpha_n
def forward(self, x):
        # x shape (batch, time_step, input_size)
        # lout shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        output, (final_hidden_state, final_cell_state) = self.lstm(x)
        output = output.permute(1, 0, 2)  # [batch, seq_len, hidden_dim*2]

        query = self.dropout(output)
        # 加入attention机制
        attn_output, alpha_n = self.attention_net(output, query)

        logit = self.fc(attn_output)

        return logit


path =  '../new/1/5/added_22.xlsx'
getd = get_DataSet(path)
train_loader,test_loader = getd.dataset_load()
