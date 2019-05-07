import numpy as np
import pandas as pd
'''
最后一步，测试，输出查准率和查全率
'''
#事件发生的时间点
event =np.array([158,219,242,250,259,287,333,352,379,403,413,423,433,439,444,452,462,499,509,519,531,533,543,
         552,563,601,603,608,627,628,634,678,679,697,698,726,737,747,748,756,765,775,789,817,836,842,860,862,
         893,994,1551,1570,1610,1813,1832,1897,1974,1994,2014,2037,2046,2062,2102,2106,2110,2117,2124,2133,
         2136,2141,2170,2186,2200,2219])
print(event.shape)

def time_in(second):
    m = int(second/60)
    s = second%60
    time = str(m)+':'+str(s)
    return time

np.set_printoptions(threshold = 1e6)
pd.set_option('display.width', None)  # 设置字符显示宽度
pd.set_option('display.max_rows', None)  # 设置显示最大行

label_pre = np.loadtxt("data_nor.txt")
# print(label_pre)
length = len(label_pre)
n = 0
time = []#time记录所有滑动1s的15s窗口，2250个窗口总计
TP = 0
F = 0
T = 0
TP2 = 0

while n < (length-14):
    time.append(label_pre[n:n+14])
    n += 1
interval = []#time中投票数达到的序列窗口号码，从1开始记，代表的时间为i~i+14
#标签0为有事件，1为无事件
#检测出来后，直接跳过
num = 0
jump =0
for ti in time:
    k = 0
    num += 1
    if jump > 0:
        jump -= 1
        continue
    for img in ti:
        if img == 0:
            k +=1
    if k > 9:
        interval.append(num)
        jump = 15
T = 0
conti_begin = []
conti_end = []
mark = []
for i in interval:
    ma = 0
    T += 1
    begin = i + 8
    end = i + 23
    print("Time interval:", time_in(begin), "--", time_in(end))
    conti_begin.append(begin)
    conti_end.append(end)
    for k in event:
        if k in range(begin+1,end+1):
            TP2 += 1
            ma = 1
            mark.append(k)
            print(k)
    if ma == 1:
        TP+=1
error = list(set(event)-set(mark))
# print('检测出来的',mark)
print('没检测出来的时间点',error)

print("T:",T,"TP:",TP)
F = int(event.shape[0])
print("F:",F)

#查全率recall=正确地标记为正TP/正确地标记为正TP+错误地标记为负FN（即原来是正，标记成负）=检测正确的事件发生/所有事件发生
#            =包含正确时间点的时间区间的数量/检测出来的所有区间的数量
#查准率 precision=正确地标记为正TP/正确地标记为正TP+错误地标记为正FP（即原来是负，标记成正）=检测正确的事件发生/所有检测标记为正
#                =时间点被准确包括的数量/所有时间点的数量

precision =TP/T
recall = TP2/F
print("precision",precision)
print("recall",recall)

#在序列中寻找连续片段
# pos=0
# conti_begin = []
# conti_end = []
# for i in range(len(interval)):
#     begin =0
#     end = 0
#     for j in range(pos+1,len(interval)):
#         if interval[j]-interval[pos] == j-pos:
#             begin = pos
#             end = j
#         else:
#             if end != 0:
#                 conti_begin.append(begin)
#                 conti_end.append(end)
#             pos = j
#             break
#     if j == len(interval)-1:
#         if end != 0:
#             conti_begin.append(begin)
#             conti_end.append(end)
#         break
#
# time_begin = []
# tiem_end = []
# for a, b in six.moves.zip(conti_begin, conti_end):
#     print(a,b)
#

# for i in range(len(interval)):
#     if i in range(a + 1, b + 1):
#         continue
#     for a, b in six.moves.zip(conti_begin, conti_end):
#         if i == a:
#             begin = interval[a]+7
#             end = interval[b]+22
#             break
#         else:
#             begin = interval[i] +7
#             end = interval[i] + 22
#         time_begin.append(begin)
#         tiem_end.append(end)
#     T += 1
#     print("Time interval:",begin,"--",end)
# for k in event:
#     for begin, end in six.moves.zip(time_begin, tiem_end):
#         mark = 0
#         if k in range(begin,end):
#             TP += 1
#             mark = 1
#             break
#     if mark == 0:
#         FN += 1
















