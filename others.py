from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import  torch
from torch.utils.data import TensorDataset,DataLoader,Dataset
from torch import nn
import NN
from LSTM import np_count
from sklearn import datasets       #导入数据模块
from sklearn.model_selection import train_test_split     #导入切分训练集、测试集模块
from sklearn.neighbors import KNeighborsClassifier
import sklearn.linear_model as lm

def count_R(pred,real):
    results = pred-real
    P = sum(real)
    TOTAL = len(real)
    N = TOTAL - P
    TP = P - np_count(results,-1)
    TN = N - np_count(results,1)
    TPR = TP / P
    TNR = TN /N
    ACC = np_count(results,0) / TOTAL
    return ACC,TPR,TNR
def BPR(x,y):
    '''
        ACC: 0.8666666666666666
        TPR: 0.909090909090909
        TNR: 0.8216867469879517

    :param x:
    :param y:
    :return:
    '''
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, )
    X = torch.from_numpy(X_train)
    X = X.to(torch.float32)
    Y = torch.from_numpy(y_train)
    Y = Y.to(torch.long)
    xtest = torch.from_numpy(X_test).to(torch.float32)
    # ytest = torch.from_numpy(y_test)
    model = NN.BPNN()
    weight_ce = torch.FloatTensor([1, 1])
    criterion = torch.nn.CrossEntropyLoss(weight=weight_ce)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    ac, rc, norc = [], [], []
    for i in range(10):
        # model.parameters()自动完成参数的初始化操作
        loss_sum = []
        # training cycle forward, backward, update

        for epoch in range(300):
            y_pred = model(X)  # forward 算预测值
            # print(y_pred,Y)
            loss = criterion(y_pred, Y)  # forward: 算损失值
            # if ((epoch+1)%50==0):
            #     print(epoch+1, loss.item())
            loss_sum.append(loss)
            optimizer.zero_grad()  # 清除上一轮的梯度，防止累积
            loss.backward()  # backward: autograd，自动计算梯度，反向传播
            optimizer.step()  # update 参数，即更新w和b的值

        model = model.eval()
        pred = model(xtest)
        out1 = pred.data.numpy()
        predictions = np.argmax(out1, axis=1)
        ACC, TPR, TNR = count_R(predictions, y_test)
        ac.append(ACC)
        rc.append(TPR)
        norc.append(TNR)
    print("ACC:", np.average(ac))
    print("TPR:", np.average(rc))
    print("TNR:", np.average(norc))
def SVMR(x,y):
    ac, rc, norc = [], [], []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,)
        # print(X_train.shape)
        # print(y_test.shape)

        clf = svm.SVC(kernel='rbf')  # SVM模块，svc,线性核函数 #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
        '''rbf
        ACC: 0.7274853801169591
        TPR: 0.844801296436952  
        TNR: 0.6148859446811433
        '''
        '''linear
        ACC: 0.7467836257309941
        TPR: 0.854808203801561
        TNR: 0.6361358725606039
        '''
        '''poly
        ACC: 0.7637426900584796
        TPR: 0.8077323375859778
        TNR: 0.7205151605866638
        '''
        '''sigmoid
        ACC: 0.4128654970760234
        TPR: 0.4089652388183757
        TNR: 0.4179597273893039
        '''
        '''precomputed
        
        '''
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        ACC, TPR, TNR = count_R(predictions, y_test)
        ac.append(ACC)
        rc.append(TPR)
        norc.append(TNR)
    print("ACC:", np.average(ac))
    print("TPR:", np.average(rc))
    print("TNR:", np.average(norc))
def logist(x,y):
    ac, rc, norc = [], [], []
    '''
    ACC: 0.7426900584795322
    TPR: 0.8158494439262196
    TNR: 0.6733709991906703

    '''
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, )
        # print(X_train.shape)
        # print(y_test.shape)
        model = lm.LogisticRegression(max_iter=300)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        ACC, TPR, TNR = count_R(predictions, y_test)
        ac.append(ACC)
        rc.append(TPR)
        norc.append(TNR)
        # 可见只有一个没有预测正确
    print("ACC:", np.average(ac))
    print("TPR:", np.average(rc))
    print("TNR:", np.average(norc))

def RNNS(x1,y):
    '''
    ACC: 0.9064327485380117
TPR: 0.9101123595505618
TNR: 0.9024390243902439
    :param x1:
    :param y:
    :return:
    '''
    INPUT_SIZE = height - 1  # 每 步输入值 / 特征数
    BATCH_SIZE = 1
    EPOCH = 200
    ac, rc, norc = [], [], []
    for i in range(10):
        print(x1.shape, y.shape)
        X_train, X_test, y_train, y_test = train_test_split(x1, y, test_size=0.3, )
        X_train = X_train.reshape(-1, BATCH_SIZE, INPUT_SIZE)
        y_train = y_train.reshape(-1, BATCH_SIZE, 1)
        X_test = X_test.reshape(-1, BATCH_SIZE, INPUT_SIZE)
        X = torch.from_numpy(X_train)
        X = X.to(torch.float32)
        Y = torch.from_numpy(y_train)
        Y = Y.to(torch.long)
        xtest = torch.from_numpy(X_test).to(torch.float32)
        # print(X_train.shape)
        # print(y_test.shape)
        model = NN.Rnn(INPUT_SIZE)

        weight_ce = torch.FloatTensor([1, 1])
        loss_func = torch.nn.CrossEntropyLoss(weight=weight_ce)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        ac, rc, norc = [], [], []
        loss_sum = []
        dataset_train = TensorDataset(X, Y)
        dataset_train = dataset_train
        train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)
        for epoch in range(EPOCH):
            for step, (x, b_y) in enumerate(train_loader):  # gives batch data
                b_x = x.view(-1, BATCH_SIZE, INPUT_SIZE)  # reshape x to (batch, time_step, input_size)
                b_x = b_x.to(torch.float32)

                output = model(b_x)  # rnn output

                # output = torch.max(output, 1)
                b_y = b_y.view(-1)
                # print(b_y)
                # b_y = b_y.squeeze(1).long()

                # print("b_y的shape：",b_y.size())   # torch.Size([1, 1, 1])
                # print(b_x.size())   # torch.Size([1, 1, 46])
                # print(output)   # torch.Size([1, 2])

                loss = loss_func(output, b_y)  # cross entropy loss
                # if ((epoch + 1) % 50 == 0 and step % 50 == 0):
                #     print(epoch + 1, loss.item())
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

        model = model.eval()
        pred = model(xtest)
        out1 = pred.data.numpy()
        predictions = np.argmax(out1, axis=1)

        ACC, TPR, TNR = count_R(predictions, y_test)
        ac.append(ACC)
        rc.append(TPR)
        norc.append(TNR)
        # 可见只有一个没有预测正确
    print("ACC:", np.average(ac))
    print("TPR:", np.average(rc))
    print("TNR:", np.average(norc))
# 数据
data_path = 'new/1/5/added_%d.xlsx' % (22)  # LSTM_DATA/1.xlsx
# data_path = '7/added_%d.xlsx' % (1)  # LSTM_DATA/1.xlsx
# 读取文件
df = pd.read_excel(data_path)
data = df.to_numpy()
height,width = data.shape
# print(height)
x1 = data[0:height - 1, :]
x1 = np.transpose(x1)
y = data[-1,:]

#
# SVMR(x1,y)
# logist(x1,y)
# BPR(x1,y)
RNNS(x1,y)