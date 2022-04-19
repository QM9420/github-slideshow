import pandas as pd
import numpy as np
from LSTM import *
import  torch
from torch.utils.data import TensorDataset,DataLoader,Dataset
from torch import nn

# 数据
data_path = 'new/1/5/added_%d.xlsx' % (22)  # LSTM_DATA/1.xlsx
# data_path = '7/added_%d.xlsx' % (1)  # LSTM_DATA/1.xlsx
# 读取文件
df = pd.read_excel(data_path)
data = df.to_numpy()
height,width = data.shape
# print(height)

# Hyper Parameters
EPOCH = 300  # 训练整批数据多少次
BATCH_SIZE = 1
TIME_STEP =  width   #  时间 步 数 / 样本数
INPUT_SIZE = height-1      #  每 步输入值 / 特征数
NAME = 'test23_LR'
HIDDEN_SIZE = 36 # 隐藏层
NUM_LAYER = 1     # LSTM层数
LR = 0.007        # l earning rate
# LRS = [0.003,0.007,0.01,0.03,0.07]
ACCS,accs,ACCS0,accs0 = [],[],[],[]
rcs,RCS,rcs0,RCS0 = [],[],[],[]
nroc,NROC,nroc0,NROC0 = [],[],[],[]
LS_loss = []
LOSS = np.ones((EPOCH))
k30 = 0
for k in range(1):
    # if k != 1:
    #     continue
    # LR=LRS[k]
    # if k<21:
    #     NUM_LAYER = 1
    #     HIDDEN_SIZE = 20 + k * 2
    # elif 21 <= k < 42:
    #     NUM_LAYER = 2
    #     HIDDEN_SIZE = 20 + (k-21) * 2
    # elif k>=42:
    #     NUM_LAYER = 3
    #     HIDDEN_SIZE = 20 + (k - 42) * 2

    for i in range(10):
    #训练数据和测试数据

        data = data_shuffle(data)


        # print("INPUTSIZE",height-1)

        dataset_x,dataset_y = create_dataset(data)
        # dataset_x, dataset_y,dataset_target = create_dataset1(data)
        # print(sum(dataset_y))  #102个负样本

        # print("width",width)
        # leny = len(dataset_y)
        num_train = int(width * 0.7)
        num_test = width - num_train
        # num_trainy = leny * 0.7


        train_x = dataset_x[:num_train]
        train_y = dataset_y[:num_train]
        train_num = len(train_y)
        # train_t = dataset_target[:num_train]

        test_x = dataset_x[num_train:]
        test_y = dataset_y[num_train:]
        # test_t = dataset_target[num_train:]

        '''
        train_x = dataset_x[num_test:]
        train_y = dataset_y[num_test:]
        
        test_x = dataset_x[:num_test]
        test_y = dataset_y[:num_test]
        '''
        # print("测试集样本标签：",test_t)
        print("测试集数量",test_y.shape)   # 131     131   44
        print("测试集中负样本数量",sum(test_y))      # 61     6    23
        test_num = len(test_y)
        # print("test_num",test_num)  # 131
        # print("train_num",num_train)  # 304
        #转换为模型的形式 （batchsize,timestep,inputsize）
        train_x = train_x.reshape(-1,BATCH_SIZE,INPUT_SIZE)
        y1 = train_y
        train_y = train_y.reshape(-1,BATCH_SIZE,1)
        train_x = torch.from_numpy(train_x)
        x1 = train_x
        x1 = x1.to(torch.float32)
        train_y = torch.from_numpy(train_y)

        train_y = train_y.to(torch.long)

        test_x = test_x.reshape(-1,BATCH_SIZE,INPUT_SIZE)
        test_x = torch.from_numpy(test_x)
        test_x = test_x.to(torch.float32)
        # test_y = test_y.reshape(-1,BATCH_SIZE,1)
        # test_y = torch.from_numpy(test_y)
        # test_y = test_y.to(torch.long)
        # test_y = test_y.view(-1)

        dataset_train = TensorDataset(train_x,train_y)
        dataset_train = dataset_train
        train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)

        model = LSTM_classifier(INPUT_SIZE,HIDDEN_SIZE,NUM_LAYER)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all parameters

        weight_ce = torch.FloatTensor([1,1])
        loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted



        # training and testing
        for epoch in range(EPOCH):
            joker = 0
            for step, (x, b_y) in enumerate(train_loader):   # gives batch data
                b_x = x.view(-1, BATCH_SIZE, INPUT_SIZE)   # reshape x to (batch, time_step, input_size)
                b_x = b_x.to(torch.float32)
                # print(b_x.size())
                output = model(b_x)               # rnn output

                # output = torch.max(output, 1)
                b_y = b_y.view(-1)
                # print(b_y)
                # b_y = b_y.squeeze(1).long()

                # print("b_y的shape：",b_y.size())   # torch.Size([1, 1, 1])
                # print(b_x.size())   # torch.Size([1, 1, 46])
                # print(output)   # torch.Size([1, 2])

                loss = loss_func(output, b_y)   # cross entropy loss


                optimizer.zero_grad()           # clear gradients for this training step
                loss.backward()                 # backpropagation, compute gradients
                optimizer.step()                # apply gradients

                LS_loss.append(loss.data.numpy())


                if (epoch+1 )% 100 == 0 and (step+1) % 100 == 0:
                #     print(joker)
                    print('Epoch: ', epoch+1, '| train loss: %.4f' % loss.data.numpy())

            LOSS[epoch] = np.average(LS_loss)
            LS_loss = []
        model = model.eval()

        out0 = model(x1)
        out0 = out0.data.numpy()
        prediction0 = np.argmax(out0,axis=1)
        results0 = prediction0 - y1
        P0= sum(y1)
        TOTAL0 = y1.shape
        N_TOTAL0 = TOTAL0 - P0
        TN0 = N_TOTAL0 - np_count(results0, 1)
        TNR0 = TN0 / N_TOTAL0  # 负样本正确率
        nroc0.append(TNR0)
        NROC0.append(TNR0)

        TP0 = P0 - np_count(results0, -1)  # 正样本的正确率误率
        RC0 = TP0 / P0
        rcs0.append(RC0)
        RCS0.append(RC0)
        accuracy0 = sum(prediction0 == y1) / train_num

        accs0.append(accuracy0)
        ACCS0.append(accuracy0)



        out1 = model(test_x)
        out1 = out1.data.numpy()
        # print(out1)
        # print(out1.shape)
        prediction = np.argmax(out1,axis=1)

        print(prediction)

        # print(test_y)
        results = prediction-test_y
        print(results)

        P = sum(test_y)
        TOTAL = test_y.shape
        N_TOTAL = TOTAL-P
        TN = N_TOTAL - np_count(results,1)
        TNR = TN/N_TOTAL   # 负样本正确率
        nroc.append(TNR)
        NROC.append(TNR)

        TP = P - np_count(results,-1)  #正样本的正确率误率
        RC = TP/P
        rcs.append(RC)
        RCS.append(RC)
        print("测试负样本：",sum(test_y))
        print("错误：",sum(np.abs(results)))
        print(prediction.shape)
        # w0,w1,w2 = check_wrong(test_t,results,w0,w1,w2)

        accuracy = sum(prediction == test_y) / test_num

        accs.append(accuracy)
        ACCS.append(accuracy)
        write_excel8(LOSS,i, 'loss', NAME)
        LOSS = np.ones((EPOCH))

        if((i+1) % 10 == 0):
            print("HIDDEN_SIZE:",HIDDEN_SIZE)
            print("accuracy:",np.average(accs))
            print("RC",np.average(rcs))
            print("TNR",np.average(nroc))
            accs = []
            rcs = []
            nroc = []
    # w0 = np.array(w0)
    # w1 = np.array(w1)
    # w2 = np.array(w2)
    ACCS = np.array(ACCS)
    RCS = np.array(RCS)
    NROC = np.array(NROC)
    ACCS0 = np.array(ACCS0)
    RCS0 = np.array(RCS0)
    NROC0 = np.array(NROC0)
    # write_excel4(w0, 0)
    # write_excel4(w1, 1)
    # write_excel4(w2, 2)
    # k = 50 + 1 + k
    write_excel9(ACCS0,NUM_LAYER,k,'accs','t'+NAME)
    write_excel9(RCS0 ,NUM_LAYER,k,'rcs','t'+NAME)
    write_excel9(NROC0,NUM_LAYER,k,'nroc','t'+NAME)
    # write_excel7(LOSS,i,'loss')

    write_excel9(ACCS, NUM_LAYER,k, 'accs',NAME)
    write_excel9(RCS,NUM_LAYER, k, 'rcs',NAME)
    write_excel9(NROC, NUM_LAYER,k, 'nroc',NAME)

    RCS,RCS0= [],[]
    ACCS,ACCS0= [],[]
    NROC,nroc,NROC0,nroc0 = [],[],[],[]
    rcs,accs,rcs0,accs0 = [],[],[],[]
    # print(EPOCH)


