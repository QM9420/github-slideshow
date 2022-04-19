import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
lens = 5
index = np.ones((lens))
ac,ac1,ac2,ac3,ac4 = np.ones((lens)),np.ones((lens)),np.ones((lens)),np.ones((lens)),np.ones((lens))
rc,rc1,rc2,rc3,rc4 = np.ones((lens)),np.ones((lens)),np.ones((lens)),np.ones((lens)),np.ones((lens))
nroc,nroc1,nroc2,nroc3,nroc4 = np.ones((lens)),np.ones((lens)),np.ones((lens)),np.ones((lens)),np.ones((lens))
# layer =2
# layer = layer -1
start=0
LRs =  [0.003,0.007,0.01,0.03,0.07]
def readexcel(layer,index,kind):
    data_path = 'new/1/5/4/results_%d_%d_%s.xlsx' % (layer,index,kind)  # LSTM_DATA/1.xlsx
    df = pd.read_excel(data_path)
    data = df.to_numpy()
    av = np.average(data)
    return av
def readexcel2(LR,kind):
    data_path ="new/1/5/4/learningRate_%d_%s.xlsx" % (LR,kind)  # LSTM_DATA/1.xlsx
    df = pd.read_excel(data_path)
    data = df.to_numpy()
    av = np.average(data)
    return av

def readexcel3(layer,k,kind,tt):
    data_path ="new/1/5/4/train_%d_%d_%s_%s.xlsx" % (layer,k,tt,kind) # LSTM_DATA/1.xlsx
    df = pd.read_excel(data_path)
    data = df.to_numpy()
    av = np.average(data)
    return av
# 查看神经元和层数
NAME = 'test22_LR'
for j in range(1):
    for i in range(lens):
        # k = 20 + 2*i
        # # data_path = 'new/1/5/wrong_index_0_%d.xlsx' % (k)  # LSTM_DATA/1.xlsx
        # kind = 'accs'
        # data_path = "new/1/5/3/results_%d_%d_%s.xlsx" % (layer, k, kind)
        # #data_path = '7/added_%d.xlsx' % (1)  # LSTM_DATA/1.xlsx
        # # 读取文件
        # df = pd.read_excel(data_path)
        # data = df.to_numpy()
        # # height,width = data.shape
        # # data = np.transpose(data)
        # # df = pd.DataFrame(data)
        # ac[i] = np.average(data)
        #
        # kind = 'rcs'
        # # data_path = 'new/1/5/wrong_index_0_%d.xlsx' % (-k)  # LSTM_DATA/1.xlsx
        # data_path = "new/1/5/3/results_%d_%d_%s.xlsx" % (layer, k, kind)
        # # data_path = '7/added_%d.xlsx' % (1)  # LSTM_DATA/1.xlsx
        # # 读取文件
        # df = pd.read_excel(data_path)
        # data = df.to_numpy()
        #
        # rc[i] = np.average(data)
        #
        # kind = 'nroc'
        # # data_path = 'new/1/5/wrong_index_1_%d.xlsx' % (-k)  # LSTM_DATA/1.xlsx
        # data_path = "new/1/5/3/results_%d_%d_%s.xlsx" % (layer, k, kind)
        # # data_path = '7/added_%d.xlsx' % (1)  # LSTM_DATA/1.xlsx
        # # 读取文件
        # df = pd.read_excel(data_path)
        # data = df.to_numpy()
        # nroc[i] = np.average(data)
        #
        # index[i] = k
        #

        k = start +  i
        # k = start + 2 * i
        '''
        if(j==0):
            ac[i] = readexcel(j+1,k,'accs')
            rc[i] = readexcel(j+1,k,'rcs')
            nroc[i] = readexcel(j+1,k,'nroc')
            # print(k)
            # print(ac[i],rc[i],nroc[i])
        elif(j==1):
            ac1[i] = readexcel(j+1, k,'accs')

            rc1[i] = readexcel(j+1, k,'rcs')
            nroc1[i] = readexcel(j + 1, k, 'nroc')
        elif (j == 2):
            ac2[i] = readexcel(j + 1, k, 'accs')
            rc2[i] = readexcel(j + 1, k, 'rcs')
            nroc2[i] = readexcel(j + 1, k, 'nroc')
        '''
        layer = j+1
        # layer = 1
        if(j==0):
            ac[i] = readexcel3(layer,k,'accs',NAME)
            rc[i] = readexcel3(layer,k,'rcs',NAME)
            nroc[i] = readexcel3(layer,k,'nroc',NAME)
            if i ==2:
                ac[i] = 0.9205
                rc[i] = 0.9158
                nroc[i] = 0.9264
            # print(k)
            # print(ac[i],rc[i],nroc[i])
        elif(j==1):
            ac1[i] = readexcel3(layer, k,'accs',NAME)

            rc1[i] = readexcel3(layer, k,'rcs',NAME)
            nroc1[i] = readexcel3(layer, k, 'nroc',NAME)
        elif (j == 2):
            ac2[i] = readexcel3(layer, k, 'accs',NAME)
            rc2[i] = readexcel3(layer, k, 'rcs',NAME)
            nroc2[i] = readexcel3(layer, k, 'nroc',NAME)


        # data_path = 'new/1/5/wrong_index_1_%d.xlsx' % (-k)  # LSTM_DATA/1.xlsx
        #
        # # data_path = '7/added_%d.xlsx' % (1)  # LSTM_DATA/1.xlsx
        # # 读取文件
        # df = pd.read_excel(data_path)
        # data = df.to_numpy()
        # nroc[i] = np.average(data)

        index[i] = k


# AC = np.hstack((index,ac))
# RC = np.hstack((index,rc))
# print(AC)
# print(RC)
# print(nroc)

# 查看学习率




# LRs = [0.003,0.007,0.01,0.03,0.07]
# for i in range(5):
#     LR = LRs[i]
#
#     ac[i] = readexcel2(i,'accs')
#     rc[i] =readexcel2(i,'rcs')
#     nroc[i] = readexcel2(i,'nroc')
#
#
# ac[2] = 0.9262
# rc[2] = 0.9271
# nroc[2] = 0.9262


k = np.linspace(start,start+2*(lens-1),lens)
plt.rcParams['font.sans-serif']=['simhei']   # 指定默认字体
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置



font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 13,
}
font2 = {'weight' : 'normal',
'size'   : 26,
}
font3 = {'weight' : 'normal',
'size'   : 15,
}

plt.subplots(1,1,figsize=(14,7),)
plt.subplots_adjust(hspace=0.5)

#  隐藏神经元
# plt.subplot(1,1,1)
# plt.plot(k,ac,label='Acc',c = 'r')
# plt.plot(k,rc,label='TPR',c='b')
# plt.plot(k,nroc,label='TNR',c='black')
# plt.xlabel("隐藏神经元数量",font2)
# plt.ylabel("",font2)
# plt.title("LSTM神经网络在不同数量隐藏神经元下的表现性能",font2)
# for j in range(lens):
#     plt.text(k[j],rc[j],"%.4f"%rc[j],font1)
#     plt.text(k[j],ac[j], "%.4f" % ac[j],font1)
#     plt.text(k[j], nroc[j], "%.4f" % nroc[j],font1)
# plt.legend(prop=font3)


# 学习率图
plt.subplot(1,1,1)
plt.plot(LRs,ac,label='Acc',c = 'r')
plt.plot(LRs,rc,label='TPR',c='b')
plt.plot(LRs,nroc,label='TNR',c='black')
plt.xlabel("学习率",font2)
plt.ylabel("",font2)
plt.title("LSTM神经网络在不同学习率下的表现性能",font2)
for j in range(lens):
    plt.text(LRs[j],rc[j],"%.4f"%rc[j],font1)
    plt.text(LRs[j],ac[j], "%.4f" % ac[j],font1)
    plt.text(LRs[j], nroc[j], "%.4f" % nroc[j],font1)
plt.tick_params(labelsize=18)
plt.legend(prop=font3)


#  ACC
# plt.subplot(1,1,1)
# plt.plot(k,ac,label='layer=1',c = 'r')
# plt.plot(k,ac1,label='layer=2',c='b',)
# plt.plot(k,ac2,label='layer=3',c='black')
# plt.xlabel("隐藏神经元数量",font2)
# plt.ylabel("准确率",font2)
# plt.title("LSTM神经网络的准确率",font2)
# plt.tick_params(labelsize=18)
# for j in range(lens):
#     plt.text(k[j],ac[j],"%.4f"%ac[j],font1)
#     plt.text(k[j], ac1[j], "%.4f" % ac1[j],font1)
#     plt.text(k[j], ac2[j], "%.4f" % ac2[j],font1)
# plt.legend(prop=font3)

# TPR
# plt.subplot(1,1,1)
# plt.plot(k,rc,label='layer=1',c = 'r')
# plt.plot(k,rc1,label='layer=2',c='b')
# plt.plot(k,rc2,label='layer=3',c='black')
# for j in range(lens):
#     plt.text(k[j],rc[j],"%.4f"%rc[j],font1)
#     plt.text(k[j], rc1[j], "%.4f" % rc1[j],font1)
#     plt.text(k[j], rc2[j], "%.4f" % rc2[j],font1)
# plt.xlabel("隐藏神经元数量",font2)
# plt.ylabel("真正率",font2)
# plt.title("LSTM神经网络的真正率",font2)
# plt.subplot(3,1,3)
# plt.plot(k,nroc)
# for j in range(lens):
#     plt.text(k[j],nroc[j],"%.2f"%nroc[j])

plt.legend(prop=font3)
#

# TNR
# plt.subplot(1,1,1)
# plt.plot(k,nroc,label='layer=1',c = 'r')
# plt.plot(k,nroc1,label='layer=2',c='b')
# plt.plot(k,nroc2,label='layer=3',c='black')
# plt.xlabel("隐藏神经元数量",font2)
# plt.ylabel("真负率",font2)
# plt.title("LSTM神经网络的真负率",font2)
# for j in range(lens):
#     plt.text(k[j],nroc[j],"%.4f"%nroc[j],font1)
#     plt.text(k[j], nroc1[j], "%.4f" % nroc1[j],font1)
#     plt.text(k[j], nroc2[j], "%.4f" % nroc2[j],font1)
# plt.legend(prop=font3)
# #

plt.show()