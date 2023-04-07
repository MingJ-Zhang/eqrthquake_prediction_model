from cProfile import label
from cmath import nan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




# 统计个数
# # df = pd.read_csv('./data_zr_jxz.csv')

# # print(df['jxz_51007_1_3123'][df['jxz_51007_1_3123'] >= 3.0].count())
# # print(df['jxz_51007_2_3123'][df['jxz_51007_2_3123'] >= 3.0].count())
# # print(df['jxz_53010_2_3123'][df['jxz_53010_2_3123'] >= 3.0].count())
# # print(df['jxz_53010_3_3123'][df['jxz_53010_3_3123'] >= 3.0].count())
# # print(df['zr_51007_1_3123'][df['zr_51007_1_3123'] >= 3.0].count())
# # print(df['zr_51007_2_3123'][df['zr_51007_2_3123'] >= 3.0].count())
# # print(df['zr_53010_2_3123'][df['zr_53010_2_3123'] >= 3.0].count())
# # print(df['zr_53010_3_3123'][df['zr_53010_3_3123'] >= 3.0].count())
# x = np.load('./data.npy')

# print(x[[0,1]].shape)

# data = np.load('./data.npy')
# np.nan_to_num(data, nan=-1)
# for i in data:
#     print(i[np.isnan(i)])
#     i[np.isnan(i)] = np.nanmean(i)

# print("----------------------------")
# for j in data:
#     print(j[np.isnan(j)])
# np.save('./data.npy',data)
la = np.load('./label.npy')
print(la.shape)
# l = len(tuple(set(la)))
# print(l)
_0 = 0
_1 = 1

for i in range(1, len(la)):
    # print(times[i], times[i-1])
    num = la[i]
    if num == 0:
        _0 += 1
    else:
        _1 += 1

print(_0)
print(_1)


# d = np.load('./data.npy')
# l = np.load('./label.npy')
# # d = np.expand_dims(d, axis=0)
# # d = np.array(d, dtype=np.object)
# print(d.shape)
# li = []
# for i in range(len(d)):
#     if d[i].__contains__(nan):
#         pass
#     else:
#         li.append(d[i])

# ll = np.array(li,dtype=np.object)
# from scipy import io
# io.savemat('./geo.mat', {'geo':{'trainlabels':l, 'train':li, 'testlabels':l, 'test':li}})

# a = [[1,2,3], [4,5,6], [7,8,9]]
# a = np.array(a, dtype=np.object)
# file_name = 'data.mat'
# io.savemat(file_name, {'a': a})
# # dataFile = '../../../../GTN-master/MTS_dataset/WalkvsRun/WalkvsRun.mat'
# # dataFile = './geo.mat'
# # data = io.loadmat(dataFile)
# # print((data['geo'][0][0]))