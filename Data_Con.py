import pandas as pd
import numpy as np

#写入
def write_excel1(WDate,index):
    data = pd.DataFrame(WDate)
    write_path = '5_1/WHOLE_%d.xlsx'%(index)
    writer = pd.ExcelWriter(write_path)
    # header参数表示列的名称，index表示行的标签
    data.to_excel(writer,'page_1',float_format='%.6f',index=False)
    writer.save()
    writer.close()
    print('%d文件已完成写入'%index)

#写入
def write_excel4(WDate,index):
    data = pd.DataFrame(WDate)
    write_path = 'new/1/5/WHOLE_%d.xlsx'%(index)
    writer = pd.ExcelWriter(write_path)
    # header参数表示列的名称，index表示行的标签
    data.to_excel(writer,'page_1',float_format='%.6f',index=False)
    writer.save()
    writer.close()
    print('%d文件已完成写入'%index)

#合成单年数据
def single_data():
    s_data = []
    for i in range(1,63):


        # 合成路径`
        data_path = 'new/1/5/%d.xlsx' % (i)  # LSTM_DATA/1.xlsx

        # 读取文件
        df = pd.read_excel(data_path)
        data = df.to_numpy()

        # 转换格式
        print(data.shape)
        height, width = data.shape
        data = data.astype(float)

        # 组合数据
        if(i == 1):
            s_data = data
        else:
            s_data = np.hstack((s_data,data))
    print(s_data.shape)
    write_excel4(s_data,1)

#合成多年数据
def multi_data():
    d_data = []
    target = []
    for i in range(1,63):

        # 合成路径`
        data_path = 'new/1/5/%d.xlsx' % (i)  # LSTM_DATA/1.xlsx

        # 读取文件
        df = pd.read_excel(data_path)
        data = df.to_numpy()

        # 转换格式
        # print(data.shape)
        height, width = data.shape
        data = data.astype(float)

        # 组合数据
        for j in range(0,width-1):
            if(i ==1 and j == 0):
                d_data = data[0:height-1,j]
                d_data = d_data.reshape(-1,1)
                d_data = np.vstack((d_data,data[0:height-1,j + 1 ].reshape(-1,1)))
                target = data[-1,j]
                # print(d_data)
                # print(target)
            else:
                temp = data[0:height-1,j]
                temp = temp.reshape(-1,1)
                temp = np.vstack((temp,data[0:height-1,j+1].reshape(-1,1)))
                target = np.hstack((target,data[-1,j]))
                d_data = np.hstack((d_data,temp))
                # print(d_data.shape)
                # print(target.shape)
        # WHOLE_Data = np.vstack((d_data,target))
        # print(WHOLE_Data.shape)
        # print(d_data.shape)
        # print(target.shape)
    target = target.reshape(1,-1)
    WHOLE_Data = np.vstack((d_data,target))
    print(WHOLE_Data.shape)
    write_excel4(WHOLE_Data,2)

#合成多年数据
def multi_data3():
    d_data = []
    target = []
    for i in range(1,63):


        # 合成路径`

        data_path = 'new/1/5/%d.xlsx' % (i)  # LSTM_DATA/1.xlsx

        # 读取文件
        df = pd.read_excel(data_path)
        data = df.to_numpy()

        # 转换格式
        # print(data.shape)
        height, width = data.shape
        data = data.astype(float)

        # 组合数据
        for j in range(0,width-2):
            if(i ==1 and j == 0):
                d_data = data[0:height-1,j]
                d_data = d_data.reshape(-1,1)
                d_data = np.vstack((d_data,data[0:height-1,j + 1 ].reshape(-1,1)))
                d_data = np.vstack((d_data, data[0:height-1, j + 2].reshape(-1, 1)))
                target = data[-1,j]
                # print(d_data)
                # print(target)
            else:
                temp = data[0:height-1,j]
                temp = temp.reshape(-1,1)
                temp = np.vstack((temp,data[0:height-1,j+1].reshape(-1,1)))
                temp = np.vstack((temp, data[0:height - 1, j + 2].reshape(-1, 1)))
                target = np.hstack((target,data[-1,j]))
                d_data = np.hstack((d_data,temp))
                # print(d_data.shape)
                # print(target.shape)
        # WHOLE_Data = np.vstack((d_data,target))
        # print(WHOLE_Data.shape)
        # print(d_data.shape)
        # print(target.shape)
    target = target.reshape(1,-1)
    WHOLE_Data = np.vstack((d_data,target))
    print(WHOLE_Data.shape)
    write_excel4(WHOLE_Data,3)

single_data()
multi_data()
multi_data3()

