from random import shuffle
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import shutil
import pandas as pd

'''
第二步 存入HDF5文件
标签不同的图片集需要手动分类
'''
shuffle_data = True
# shuffle the addresses before saving
###########################################################################

#颜色归一化映射后的总测试集H5文件（没有标签，顺序正确，用来输出时间段）
hdf5_path ='E:/Graduation-Project/Event&NoEvent/HDF5/dataset506_all.hdf5'
###########################################################################

#训练集的分了类的，有标签的
train_path = 'E:/Graduation-Project/Event&NoEvent/HDF5/dataset112_val.hdf5'
#归一化前的测试集分了类的，用来测试准确率
test_path2 = 'E:/Graduation-Project/Event&NoEvent/HDF5/test_416.hdf5'
#归一化后的测试分类的
test_path = 'E:/Graduation-Project/Event&NoEvent/HDF5/dataset430_nor.hdf5'

###########################################################################
#训练图片集
event_path = 'E:/Graduation-Project/Event&NoEvent/Even_original/Even_spec_224'
NOevent_path = 'E:/Graduation-Project/Event&NoEvent/Even_original/No_event_spec_224'
#之前分好类的测试图片集
event_path2 = 'E:/Graduation-Project/Event&NoEvent/TEST_Even'
NOevent_path2 = 'E:/Graduation-Project/Event&NoEvent/TEST_No_Even'
###########################################################################
#根据之前的分类进行移动的归一化后得到的分类的数据集
path = 'E:/Graduation-Project/Event&NoEvent/TEST_Normalized'
movepath1 = 'E:/Graduation-Project/Event&NoEvent/TEST_Nor_even'
movepath2 = 'E:/Graduation-Project/Event&NoEvent/TEST_Nor_no_even'
###########################################################################
pd.set_option('display.width', 1000)  # 设置字符显示宽度
pd.set_option('display.max_rows', None)  # 设置显示最大行

# 选择全部移除或者指定后缀名文件
def movefile(dir_path,move_path,name_path):#目标文件夹（总）；移动的文件夹（标签分类）；根据该文件夹的命名来移动
    count = os.listdir(dir_path)
    count2 = os.listdir(name_path)
    i = -1
    for item in count:
        i += 1
        if item.endswith('.png'):
            src = os.path.join(os.path.abspath(dir_path), item)
            for it in count2:
                name = os.path.join(os.path.abspath(name_path), it)
                # print(src[-8:-4])
                # print(name[-8:-4])
                if src[-8:-4] == name[-8:-4]:
                    full_path = os.path.join(dir_path, item)
                    despath = move_path + '/0000' + format(str(i), '0>4s') + '.png'
                    shutil.move(full_path, despath)

#read addresses and labels from the 'train' folder
def get_files(file_dir1,file_dir2):
    boat1 = []
    label_boat1 =[]
    boat2 = []
    label_boat2 = []

    for file in sorted(os.listdir(file_dir1)):
        print(os.path.join(file_dir1, file))
        boat1.append(file_dir1+'/'+file)
        label_boat1.append(0)#事件标签为0
    for file in sorted(os.listdir(file_dir2)):
        print(os.path.join(file_dir2, file))
        boat2.append(file_dir2+'/'+file)
        label_boat2.append(1)#事件的标签为1
    #合起来组成一个list（img和lab）
    image_list = np.hstack((boat1, boat2))
    label_list = np.hstack((label_boat1, label_boat2))
    # 利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    # 从打乱的temp中再取出list（img和lab）
    image_list = temp[:, 0]
    print(len(image_list))
    label_list = temp[:, 1]
    label_list = [int(i) for i in label_list]
    return image_list, label_list
    # 返回两个list 分别为图片文件名及其标签  顺序已被打乱

def get_files1(file_dir1):
    boat1 = []
    for file in sorted(os.listdir(file_dir1)):
        # print(os.path.join(file_dir1, file))
        boat1.append(file_dir1+'/'+file)
    return boat1

#测试集2261张图
def create_HDF5(path1,path2,hdf5_path):
    image_list, label_list = get_files(path1,path2)
    print("图片集的总数量",len(image_list))
    print("标签的总数量",len(label_list))

    Train_image = np.random.rand(len(image_list) - 500, 112, 112, 3).astype('float32')
    Train_label = np.random.rand(len(image_list) - 500, 1).astype('float32')
    Val_image = np.random.rand(250, 112, 112, 3).astype('float32')
    Val_label = np.random.rand(250, 1).astype('float32')
    Test_image = np.random.rand(len(image_list), 112, 112, 3).astype('float32')
    Test_label = np.random.rand(len(image_list), 1).astype('int')

    for i in range(len(image_list) - 500):
        try:
            Train_image[i] = np.array(plt.imread(image_list[i]))
            Train_label[i] = np.array(label_list[i])
        except:
            pass
        continue

    for i in range((len(image_list) - 500), (len(image_list)-250)):
        try:
            Val_image[i + 500 - len(image_list)] = np.array(plt.imread(image_list[i]))
            Val_label[i + 500 - len(image_list)] = np.array(label_list[i])
        except:
            pass
        continue

    for i in range((len(image_list) - 250), (len(image_list))):
        try:
            Test_image[i + 250 - len(image_list)] = np.array(plt.imread(image_list[i]))
            Test_label[i + 250 - len(image_list)] = np.array(label_list[i])
        except:
            pass
        continue
    for i in range(len(image_list)):
        try:
            Test_image[i] = np.array(plt.imread(image_list[i]))
            Test_label[i] = np.array(label_list[i])
        except:
            pass
        continue
    # Create a new file
    f = h5py.File(hdf5_path, 'w')
    f.create_dataset('X_train', data=Train_image)
    f.create_dataset('y_train', data=Train_label)
    f.create_dataset('X_val', data=Val_image)
    f.create_dataset('y_val', data=Val_label)
    f.create_dataset('X_test', data=Test_image)
    f.create_dataset('y_test', data=Test_label)
    f.close()

    # Load hdf5 dataset
    dataset = h5py.File(hdf5_path, 'r')
    train_set_x_orig = np.array(dataset['X_train'][:]) # your train set features
    train_set_y_orig = np.array(dataset['y_train'][:]) # your train set labels
    val_set_x_orig = np.array(dataset['X_val'][:])
    val_set_y_orig = np.array(dataset['y_val'][:])
    test_set_x_orig = np.array(dataset['X_test'][:])
    test_set_y_orig = np.array(dataset['y_test'][:])
    f.close()
    print("训练集X的大小",train_set_x_orig.shape)
    print("训练集标签y的大小",train_set_y_orig.shape)
    # 测试
    plt.figure(1)
    plt.imshow(test_set_x_orig[233],cmap='jet')
    plt.show()
    print(test_set_y_orig[233])

def create_HDF51(data_path,hdf5_path):
    image_list = get_files(data_path)
    print("图片集的总数量",len(image_list))

    Train_image = np.random.rand(len(image_list), 112, 112, 3).astype('float32')

    for i in range(len(image_list)):
        try:
            Train_image[i] = np.array(plt.imread(image_list[i]))
        except:
            pass
        continue
    # Create a new file
    f = h5py.File(hdf5_path, 'w')
    f.create_dataset('X_test', data=Train_image)
    f.close()
    # Load hdf5 dataset
    dataset = h5py.File(hdf5_path, 'r')
    train_set_x_orig = np.array(dataset['X_test'][:]) # your train set features
    f.close()

    # 测试
    plt.figure(1)
    plt.imshow(train_set_x_orig[233],cmap='jet')
    plt.show()
#
def test():
    # Load hdf5 dataset
    dataset = h5py.File(hdf5_path, 'r')
    train_set_x_orig = np.array(dataset['X_test'][:])
    # print(test_set_x_orig.shape)
    plt.figure(1)
    plt.imshow(train_set_x_orig[19], cmap='jet')
    plt.show()

if __name__ == '__main__':
    #移动
    movefile(path, movepath1, event_path2)
    movefile(path, movepath2, NOevent_path2)
    #创建HDF5
    create_HDF5(event_path,NOevent_path,train_path)
    create_HDF5(event_path2, NOevent_path2, test_path2)
    create_HDF5(movepath1, movepath2, test_path)
    create_HDF51(path,hdf5_path)




