import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn


#导入数据
def Data_count():
    total_num = 0
    zero_num = 0
    for i in range(1,63):
        # 合成路径`
        data_path = 'H_CData3_1/%d.xlsx' % (i)  # LSTM_DATA/1.xlsx

        # 读取文件
        df = pd.read_excel(data_path)
        data = df.to_numpy()
        height,width = data.shape
        for j in range(0,height):
            for k in range(0,width):
                total_num += 1
                if data[j,k] == 0:
                    zero_num += 1
    return total_num,zero_num
'''
total,zero = Data_count()
print("数据项总数：%d"%(total))
print("缺失数据项数量：%d"%(zero))
print("数据缺失比率：%.4f"%(zero/total))

数据项总数：16434
缺失数据项数量：1413
数据缺失比率：0.0860

数据项总数：15904
缺失数据项数量：572
数据缺失比率：0.0360
'''

#数据取平均值
def Data_marke():    # 第10,11,16,17,33行不需要填补缺失值
    average_data = np.ones((32,17))
    aver_num = np.ones((32,17))
    sum_data = np.zeros((32,17))

    #读取每一年的各属性值，并计算平均值
    for i in range(1,63):
        # 合成路径`
        data_path = 'new/1/1/%d.xlsx' % (i)  # LSTM_DATA/1.xlsx

        # 读取文件
        df = pd.read_excel(data_path)
        data = df.to_numpy()

        #去掉标签
        data = data[0:32,:]
        height,width = data.shape
        data = data.astype(float)
        # print(type(data[1,1]))
        # print(type(sum_data[1,1]))
        # print(data)
        # print(data.shape)
        # print(i)
        #加和计算
        for j in range(0, height):
            for k in range(0, width):
                if(data[j,k] !=0 ):
                    sum_data[j,k] = sum_data[j,k] + data[j,k]
                    aver_num[j,k] += 1

    height,width = aver_num.shape
    for j in range(0, height):
        for k in range(0, width):

            if(aver_num[j,k] > 1):
                average_data[j,k] = sum_data[j,k] / (aver_num[j,k]-1)

    return  average_data


# average_data = Data_marke()
# path = 'new/1/1/AV_DATA.xlsx'
# data = pd.DataFrame(average_data)
# write_path = path
# writer = pd.ExcelWriter(write_path)
# # header参数表示列的名称，index表示行的标签
# data.to_excel(writer, 'page_1', float_format='%.6f', index=False)
# writer.save()
# writer.close()
# print('文件已完成写入')


def plotdata(data, index):
    x = data[index, :]
    y = len(x)
    y = np.linspace(1,y,y)
    plt.rcParams['font.sans-serif'] = ['simhei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    plt.plot(y, x, marker='o')
    for y1,x1 in zip(y,x):
        plt.text(y1,x1,x1,fontdict={'fontsize':14})
    plt.show()

# 写入函数
def write_excel(WDate,index):
    data = pd.DataFrame(WDate)
    write_path = 'new/1/2/%d.xlsx'%(index)
    writer = pd.ExcelWriter(write_path)
    # header参数表示列的名称，index表示行的标签
    data.to_excel(writer,'page_1',float_format='%.6f',index=False)
    writer.save()
    writer.close()
    print('%d文件已完成写入'%index)


# 写入函数 归一化
def write_excel1(WDate,index):
    data = pd.DataFrame(WDate)
    write_path = 'Data_5/3/%d.xlsx'%(index)
    writer = pd.ExcelWriter(write_path)
    # header参数表示列的名称，index表示行的标签
    data.to_excel(writer,'page_1',float_format='%.6f',index=False)
    writer.save()
    writer.close()
    print('%d文件已完成写入'%index)


# 数据填补
def Data_fill():  # 第10,11,16,17,33行不需要填补缺失值
    # 合成路径`
    data_path = 'new/1/1/AV_DATA.xlsx' # LSTM_DATA/1.xlsx


    # 读取文件
    df = pd.read_excel(data_path)
    av_data = df.to_numpy()

    # 转化为float型
    av_height, av_width = av_data.shape
    av_data = av_data.astype(float)

    for i in range(1, 63):
        # 合成路径`
        data_path = 'new/1/1/%d.xlsx' % (i)  # LSTM_DATA/1.xlsx

        # 读取文件
        df = pd.read_excel(data_path)
        data = df.to_numpy()

        # 去掉标签
        # print(data.shape)
        height, width = data.shape
        data = data.astype(float)
        # plotdata(data, 25)
        # print(data)

        # 数据填补
        for j in range(0,height):
            for k in range(0,width):
                if(
                    j != 9 and
                    j != 10 and
                    j != 15 and
                    j != 16 and
                    j != 32
                ):
                    if(data[j,k] == 0):
                        data[j,k] = av_data[j,k]
        write_excel(data,i)
        # print(data)
        # plotdata(data,25)
# Data_fill()
#数据交换
def sweap(a,b):
    t = a
    a = b
    b = t
    return a,b


#数据归一化
def data_t1():
    h = 32
    maxs = np.zeros((h, 1))
    mins = np.zeros((h, 1))
    i = 1
    data_path = 'Data_5/WHOLE_%d.xlsx' % (i)  # LSTM_DATA/1.xlsx
    # 读取文件
    df = pd.read_excel(data_path)
    data = df.to_numpy()

    # 转换格式
    # print(data.shape)
    height, width = data.shape
    data = data.astype(float)
    for i in range(h):
        maxs[i] = np.max(data[i, :])
        mins[i] = np.min(data[i, :])
    for j in range(h):
        data[j, :] = (data[j, :] - mins[j]) / (maxs[j] - mins[j])
    write_excel1(data, 1)
# data_t1()


def Data_to1():  #1-9 13 22 23 25 26 29 30
    h =32
    maxs = np.zeros((h,1))
    mins = np.zeros((h,1))



    for i in range(1,63):
        # 合成路径`
        data_path = 'Data_5/%d.xlsx' % (i)  # LSTM_DATA/1.xlsx

        # 读取文件
        df = pd.read_excel(data_path)
        data = df.to_numpy()

        # 转换格式
        # print(data.shape)
        height, width = data.shape
        data = data.astype(float)

        # 寻找最大值和最小值
        for j in range(0,height-1):
            for k in range(0,width):
                if (maxs[j] < data[j,k]):
                    maxs[j] = data[j,k]
                if (mins[j] > data[j,k]):
                    mins[j] = data[j,k]
    '''
    print(maxs)
    print(mins)
    fid = np.linspace(0,31,32)
    F = []

    for j in range(32):
         F.append( 'f'+np.str(np.int(fid[j])))
    F = np.array(F)
    # print(F)
    plt.rcParams['font.sans-serif'] = ['simhei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    plt.xlabel("特征变量")
    plt.ylabel("数值大小(10^0)")
    plt.title("各特征变量的最大值与最小值")
    # plt.text(time, data, data, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小
    plt.plot(F, maxs, markersize=3, alpha=1,c='b',label='最大值')
    plt.plot(F,mins,markersize=3, alpha=1,c='r',label='最小值')
    plt.legend()
    plt.show()

    '''
    for i in range(1,63):
        # 合成路径`
        data_path = 'Data_5/%d.xlsx' % (i)  # LSTM_DATA/1.xlsx

        # 读取文件
        df = pd.read_excel(data_path)
        data = df.to_numpy()

        # 转换格式
        # print(data.shape)
        height, width = data.shape

        data = data.astype(float)

        for j in range(0,height-1):
            data[j,:] = (data[j,:] - mins[j]) / (maxs[j] - mins[j])
        write_excel1(data,i)

# Data_to1()

# 数据第四次清洗
def data_del(data):   # 2 4 6 7 8 10 16 26 32

    # data = np.delete(data,[1,3,5,6,7,9,15,25,31],axis=0)
    # print(data.shape)
    # data = np.delete(data,[4,5,7,25,31],axis=0)
    data = np.delete(data, [1,4,5, 15], axis=0)
    return  data

def write_excel4(WDate,index):
    data = pd.DataFrame(WDate)
    write_path = 'new/1/5/%d.xlsx'%(index)
    writer = pd.ExcelWriter(write_path)
    # 数据写入
    data.to_excel(writer,'page_1',float_format='%.6f',index=False)
    writer.save()
    writer.close()
    print('%d文件已完成写入'%index)

def Data_select():
    for i in range(1, 63):
        # 合成路径`
        data_path = 'new/1/4/%d.xlsx' % (i)  # LSTM_DATA/1.xlsx

        # 读取文件
        df = pd.read_excel(data_path)
        data = df.to_numpy()

        # 转换格式
        # print(data.shape)
        height, width = data.shape
        data = data.astype(float)

        # 数据删除
        data = data_del(data)
        print(data.shape)
        write_excel4(data,i)

# Data_select()

def write_excel5(WDate,index):
    data = pd.DataFrame(WDate)
    write_path = 'Data_5/%d.xlsx'%(index)
    writer = pd.ExcelWriter(write_path)
    # 数据写入
    data.to_excel(writer,'page_1',float_format='%.5f',index=False)
    writer.save()
    writer.close()
    print('%d文件已完成写入'%index)


def data_select5():

    for i in range(1, 63):
        # 合成路径`
        data_path = 'H_CData3_2/%d.xlsx' % (i)  # LSTM_DATA/1.xlsx

        # 读取文件
        df = pd.read_excel(data_path)
        data = df.to_numpy()

        height,width = data.shape
        if(width<=5):
            write_excel5(data,i)

        else:
            data = data[:,0:5]
            # print(data.shape)
            write_excel5(data,i)
# data_select5()

def data_append(out,data,h,w):
    x = data[h,w]

    out.append(x)
    return  out

def conditions(x):
   return x<2400 and x>-1500



def data_classifer():  # 13 22  23 25 29 30
    data_13 = []
    data_22 = []
    data_23 = []
    data_25 = []
    data_29 = []
    data_30 = []
    for i in range(1, 63):
        # 合成路径`
        data_path = 'new/1/%d.xlsx' % (i)  # LSTM_DATA/1.xlsx

        # 读取文件
        df = pd.read_excel(data_path)
        data = df.to_numpy()

        height,width = data.shape
        for j in range(width):
            data_13  = data_append(data_13,data,12,j)
            data_22 = data_append(data_22, data, 21, j)
            data_23 = data_append(data_23, data, 22, j)
            data_25 = data_append(data_25, data, 24, j)
            data_29 = data_append(data_29, data, 28, j)
            data_30 = data_append(data_30, data, 29, j)

    lens = len(data_22)
    data = filter(conditions,data_30)
    data = list(data)
    data = np.array(data)
    av = sum(data)/len(data)
    print(av)
    data_13 = np.array(data_13)
    data_22 = np.array(data_22)
    data_23 = np.array(data_23)
    data_25 = np.array(data_25)
    data_29 = np.array(data_29)
    data_30 = np.array(data_30)

    plt.figure(figsize=(8, 4))
    seaborn.distplot(data, bins=10, hist=True, kde=False, norm_hist=False,
                 rug=True, vertical=False, label='distplot',
                 axlabel='x轴', hist_kws={'color': 'y', 'edgecolor': 'k'},
                 fit=norm)
    # 用标准正态分布拟合
    plt.legend()
    plt.grid(linestyle='--')
    plt.show()


    # print("13:%.4f,%.4f"%(max(data_13),min(data_13)))
    # print("22:%.4f,%.4f" % (max(data_22), min(data_22)))
    # print("23:%.4f,%.4f" % (max(data_23), min(data_23)))
    # print("25:%.4f,%.4f" % (max(data_25), min(data_25)))
    # print("29:%.4f,%.4f" % (max(data_29), min(data_29)))
    # print("30:%.4f,%.4f" % (max(data_30), min(data_30)))
# data_classifer()

def data_change(data,lims,zero_index=3):
    l = len(lims)
    if(l==5):
        data = np.where((data >= lims[2])&(data < lims[3]), 1, data)
        data = np.where((data >= lims[3]) & (data < lims[4]), 2, data)
        data = np.where((data >= lims[1]) & (data < lims[2]), 3, data)
        data = np.where((data >= lims[0]) & (data < lims[1]), 4, data)
        data = np.where((data >= lims[4]) , 5, data)
        data = np.where((data <= lims[0]) , 6, data)
    else:
        data = np.where((data >= lims[2]) & (data < lims[3]), 1, data)
        data = np.where((data >= lims[3]) & (data < lims[4]), 2, data)
        data = np.where((data >= lims[1]) & (data < lims[2]), 3, data)
        data = np.where((data >= lims[0]) & (data < lims[1]), 4, data)
        data = np.where((data >= lims[4]) & (data < lims[5]), 5, data)
        data = np.where((data >= lims[5]) & (data < lims[6]), 6, data)
        data = np.where((data >= lims[6]), 7, data)
        data = np.where((data <= lims[0]), 8, data)

    return data

def to_int(data):
    height,width = data.shape
    d13 = np.array([-50,-25,0,25,50,75,100])
    d22 = np.array([-100, -50, 0, 25,50, 75,100])
    d23 = np.array([-50, -25,0, 25,50,75,100])
    d25 = np.array([-400, -200, 0, 200, 400])
    d29 = np.array([-200, -100, 0, 100, 200])
    d30 = np.array([-200, -100, 0, 100, 200,300,400])

    index = [13,22,23,25,29,30]
    for i in range(height):
        if(i == 13):
            temp = data[i-1,:]
            temp = data_change(temp,d13)
            data[i-1,:] = temp
        if(i == 22):
            temp = data[i - 1, :]
            temp = data_change(temp, d22)
            data[i - 1, :] = temp
        if(i == 23):
            temp = data[i - 1, :]
            temp = data_change(temp, d23)
            data[i - 1, :] = temp
        if (i == 25):
            temp = data[i - 1, :]
            temp = data_change(temp, d25)
            data[i - 1, :] = temp
        if (i == 29):
            temp = data[i - 1, :]
            temp = data_change(temp, d29)
            data[i - 1, :] = temp
        if (i == 30):
            temp = data[i - 1, :]
            temp = data_change(temp, d30)
            data[i - 1, :] = temp
    return data


# index = np.array([13,22,23,25,29,30])
# a = np.where((index>13)&(index<23),0,index)
# print(a)
#
def write_excel6(WDate,index):
    data = pd.DataFrame(WDate)
    write_path = 'new/1/1/%d.xlsx'%(index)
    writer = pd.ExcelWriter(write_path)
    # 数据写入
    data.to_excel(writer,'page_1',float_format='%.6f',index=False)
    writer.save()
    writer.close()
    print('%d文件已完成写入'%index)


def data_toint():

    for i in range(1, 63):
        # 合成路径`
        data_path = 'new/1/%d.xlsx' % (i)  # LSTM_DATA/1.xlsx

        # 读取文件
        df = pd.read_excel(data_path)
        data = df.to_numpy()

        height,width = data.shape
        data = to_int(data)
        write_excel6(data,i)

#
# data_toint()
# if __name__ == '__main__':
    # Data_fill() #1
    # Data_to1() #2
    # Data_select()