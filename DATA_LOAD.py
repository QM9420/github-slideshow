import pandas as pd
import numpy as np
import  re
# 读取宏观数据
def load_HData():
    data_path = 'LSTM_DATA/hong.xlsx'
    df = pd.read_excel(data_path)
    Hheight,Hwidth = df.shape
    HData = df.to_numpy()
    # print(HData.shape) #18X9
    #指标名称	基础货币余额	基础货币余额:同比	消费者信心指数 	消费者预期指数	城镇居民家庭恩格尔系数	农村居民家庭恩格尔系数	货币乘数	货币和准货币(M2)

    # 数据转置，跟企业数据的形式匹配
    HData = np.transpose(HData)
    # print(HData.shape) #9X18
    # print(HData)
    # X轴 20-04
    # Y轴 基础货币余额	基础货币余额:同比	消费者信心指数 	消费者预期指数	城镇居民家庭恩格尔系数	农村居民家庭恩格尔系数	货币乘数	货币和准货币(M2)
    HData = HData[1:9,1:18]
    # print(HData.shape)  # 8X17
    # print(HData)
    return HData

#读取Excel表
def read_excel(rpath):
    df = pd.read_excel(rpath)
    rdata = df.to_numpy()
    print(rdata)
    print(rdata.shape)

# 写入函数
def write_excel(WDate,path='',index=1):
    data = pd.DataFrame(WDate)
    if(path != ''):
        write_path = path
    else:
        write_path = 'new/1/%d.xlsx'%(index)
    writer = pd.ExcelWriter(write_path)
    # header参数表示列的名称，index表示行的标签
    data.to_excel(writer,'page_1',float_format='%.5f',index=False)
    writer.save()
    writer.close()
    print('%d文件已完成写入'%index)

# 读取公司数据
def load_CData(HData):
    #宏观数据
    HWeight,HWidth = HData.shape

    for i in range(1,63):
        # 合成路径
        data_path = 'H_CData5/%d.xlsx'%(i)  # LSTM_DATA/1.xlsx

        # 读取文件
        df = pd.read_excel(data_path)
        # print(height,width)
        # print(data_path)
        # print(df)

        # 获取信息
        CHeight,CWidth = df.shape
        CData = df.to_numpy()   # 25 X N
        CData = CData[:,1:]
        # print(CData)
        # print(CData.shape)
        # print(CData[:,1])

        #截取对应年份的宏观数据
        if(i==62):
            C_Hdata = HData[:,1:CWidth]
        elif(i<62 and i>=58):
            CData = CData[:,1:]
            C_Hdata = HData[:,0:CWidth-2]
        else:
            C_Hdata = HData[:,0:CWidth-1]
        # print(C_Hdata.shape)
        # print(CData.shape)

        #将数据垂直组合
        completed_data = np.vstack((C_Hdata,CData))
        # print(completed_data.shape)  #33 X N
        # print(completed_data)
        #将数据写入
        write_excel(completed_data,'',i)

# 读写文件
HData = load_HData()
load_CData(HData)

# 检查所写文件
# for i in range(1,63):
#     path = 'new/1/%d.xlsx'%(i)
#     read_excel(path)