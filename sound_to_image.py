import soundfile as sf
import matplotlib.pyplot as plt

'''
第四步，把测试的wav声音信号转换为时频图
'''
images_dir = 'E:/Graduation-Project/Event&NoEvent/TEST_Normalized'
#15s一个窗口，一次滑动1s
#一共2278s，9112000帧
filename = 'cr_20181025-161858.wav'
'''降采样后4556000帧'''

x, samplerate = sf.read(filename)
# print(x.shape)
x = x.reshape((-1,2))
x = x.mean(axis=1)
# print(x)
for i in range(2264):
# i = 5
    fig = plt.figure(0)  # 新图 0
    spec, freqs, t, image = plt.specgram(x[(i*2000):(i*2000+30000)],NFFT=512, Fs=2000., noverlap=380,cmap='jet',vmin = -187.2,vmax = -17.6)
    plt.xticks([])
    plt.axis('off')
    plt.savefig(images_dir+'/0000' + format(str(i), '0>4s') + '.png', format='png')  # 保存
    plt.close(0)  # 关闭图 0
#输出频率，来确定最大频率和最小频率来归一化，已经确定为 -187.2 到 -17.6
    for i in freqs:
        print(i)






