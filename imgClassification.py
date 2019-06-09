import cv2
import numpy as np
 
from keras.utils import np_utils, conv_utils
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout, Dense, Activation
from keras.optimizers import Adam
 
import os
import cPickle
 
def get_name_list(filepath):                #获取各个类别的名字
    pathDir =  os.listdir(filepath)
    out = []
    for allDir in pathDir:
        if os.path.isdir(os.path.join(filepath,allDir)):
            child = allDir.decode('gbk')    # .decode('gbk')是解决中文显示乱码问题
            out.append(child)
    return out
    
def eachFile(filepath):                 #将目录内的文件名放入列表中
    pathDir =  os.listdir(filepath)
    out = []
    for allDir in pathDir:
        child = allDir.decode('gbk')    # .decode('gbk')是解决中文显示乱码问题
        out.append(child)
    return out
 
def get_data(data_name,train_percentage=0.7,resize=True,data_format=None):   #从文件夹中获取图像数据
    file_name = os.path.join(pic_dir_out,data_name+str(Width)+"X"+str(Height)+".pkl")   
    if os.path.exists(file_name):           #判断之前是否有存到文件中
        (X_train, y_train), (X_test, y_test) = cPickle.load(open(file_name,"rb"))
        return (X_train, y_train), (X_test, y_test)  
    data_format = conv_utils.normalize_data_format(data_format)
    pic_dir_set = eachFile(pic_dir_data)  
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    label = 0
    for pic_dir in pic_dir_set:
        print(pic_dir_data+pic_dir)
        if not os.path.isdir(os.path.join(pic_dir_data,pic_dir)):
            continue    
        pic_set = eachFile(os.path.join(pic_dir_data,pic_dir))
        pic_index = 0
        train_count = int(len(pic_set)*train_percentage)
        for pic_name in pic_set:
            if not os.path.isfile(os.path.join(pic_dir_data,pic_dir,pic_name)):
                continue
            img = cv2.imread(os.path.join(pic_dir_data,pic_dir,pic_name))
            if img is None:
                continue
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
            if (resize):
                img = cv2.resize(img,(Width,Height))
            if (data_format == 'channels_last'):
                img = img.reshape(-1,Width,Height,1)
            elif (data_format == 'channels_first'):
                img = img.reshape(-1,1,Width,Height)
            if (pic_index < train_count):
                X_train.append(img)
                y_train.append(label)          
            else:
                X_test.append(img)
                y_test.append(label)
            pic_index += 1
        if len(pic_set) != 0:        
            label += 1
    X_train = np.concatenate(X_train,axis=0)        
    X_test = np.concatenate(X_test,axis=0)    
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    cPickle.dump([(X_train, y_train), (X_test, y_test)],open(file_name,"wb")) 
    return (X_train, y_train), (X_test, y_test)   
 
def get_2data(data_name,resize=True,data_format=None):   #当train和test数据被分为两个部分时使用
    file_name = os.path.join(pic_dir_out,data_name+str(Width)+"X"+str(Height)+".pkl")   
    if os.path.exists(file_name):           #判断之前是否有存到文件中
        (X_train, y_train), (X_test, y_test) = cPickle.load(open(file_name,"rb"))
        return (X_train, y_train), (X_test, y_test)   
    data_format = conv_utils.normalize_data_format(data_format)
    all_dir_set = eachFile(pic_dir_data)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
 
    for all_dir in all_dir_set:
        if not os.path.isdir(os.path.join(pic_dir_data,all_dir)):
            continue
        label = 0
        pic_dir_set = eachFile(os.path.join(pic_dir_data,all_dir))
        for pic_dir in pic_dir_set:
            print(pic_dir_data+pic_dir)
            if not os.path.isdir(os.path.join(pic_dir_data,all_dir,pic_dir)):
                continue    
            pic_set = eachFile(os.path.join(pic_dir_data,all_dir,pic_dir))
            for pic_name in pic_set:
                if not os.path.isfile(os.path.join(pic_dir_data,all_dir,pic_dir,pic_name)):
                    continue
                img = cv2.imread(os.path.join(pic_dir_data,all_dir,pic_dir,pic_name))
                if img is None:
                    continue
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
                if resize:
                    img = cv2.resize(img,(Width,Height))
                if (data_format == 'channels_last'):
                    img = img.reshape(-1,Width,Height,1)
                elif (data_format == 'channels_first'):
                    img = img.reshape(-1,1,Width,Height)
                if ('train' in all_dir):
                    X_train.append(img)
                    y_train.append(label)          
                elif ('test' in all_dir):
                    X_test.append(img)
                    y_test.append(label)
            if len(pic_set) != 0:        
                label += 1
    X_train = np.concatenate(X_train,axis=0)        
    X_test = np.concatenate(X_test,axis=0)    
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    cPickle.dump([(X_train, y_train), (X_test, y_test)],open(file_name,"wb")) 
    return (X_train, y_train), (X_test, y_test)   
 
def main():
    global Width, Height, pic_dir_out, pic_dir_data
    Width = 32
    Height = 32
    num_classes = 102                   #Caltech101为102  cifar10为10
    pic_dir_out = r'E:/pic_cnn/pic_out/'  
    pic_dir_data = r'E:/pic_cnn/pic_dataset/Caltech101/'  
    (X_train, y_train), (X_test, y_test) = get_data("Caltech101_gray_data_",0.7,data_format='channels_last')
    #pic_dir_data = 'E:/pic_cnn/pic_dataset/cifar10/'
    #(X_train, y_train), (X_test, y_test) = get_2data("Cifar10_gray_data_",resize=False,data_format='channels_last')
    
    X_train = X_train/255.              #数据预处理
    X_test = X_test/255.
    print(X_train.shape)
    print(X_test.shape)
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    
    model = Sequential()                #CNN构建
    model.add(Convolution2D(
        input_shape=(Width, Height, 1),
        #input_shape=(1, Width, Height),
        filters=8,
        kernel_size=3,
        strides=1,
        padding='same',     
        data_format='channels_last',
    ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2,
        data_format='channels_last',
    ))
    model.add(Convolution2D(16, 3, strides=1, padding='same', data_format='channels_last'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2, data_format='channels_last'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        
    print('\nTraining ------------')    #从文件中提取参数，训练后存在新的文件中
    cm = 0                              #修改这个参数可以多次训练
    cm_str = '' if cm==0 else str(cm)
    cm2_str = '' if (cm+1)==0 else str(cm+1)  
    if cm >= 1:
        model.load_weights(os.path.join(pic_dir_out,'cnn_model_Caltech101_'+cm_str+'.h5'))
        #model.load_weights(os.path.join(pic_dir_out,'cnn_model_Cifar10_'+cm_str+'.h5'))    
    model.fit(X_train, y_train, epochs=10, batch_size=128,)   #正式训练数据
    model.save_weights(os.path.join(pic_dir_out,'cnn_model_Caltech101_'+cm2_str+'.h5'))
     
    print('\nTesting ------------')     #对测试集进行评估，额外获得metrics中的信息
    loss, accuracy = model.evaluate(X_test, y_test)
    print('\n')
    print('test loss: ', loss)
    print('test accuracy: ', accuracy)
    
    class_name_list = get_name_list(pic_dir_data)    #获取top-N的每类的准确率
    #class_name_list = get_name_list(os.path.join(pic_dir_data,'train'))
    pred = model.predict(X_test, batch_size=128)
    N = 5
    pred_list = []
    for row in pred:
        pred_list.append(row.argsort()[-N:][::-1])  #获取最大的N个值的下标
    pred_array = np.array(pred_list)
    test_arg = np.argmax(y_test,axis=1)
    class_count = [0 for _ in xrange(num_classes)]
    class_acc = [0 for _ in xrange(num_classes)]
    for i in xrange(len(test_arg)):
        class_count[test_arg[i]] += 1
        if test_arg[i] in pred_array[i]:
            class_acc[test_arg[i]] += 1
    print('top-'+str(N)+' all acc:',str(sum(class_acc))+'/'+str(len(test_arg)),sum(class_acc)/float(len(test_arg)))
    for i in xrange(num_classes):
        print (i, class_name_list[i], 'acc: '+str(class_acc[i])+'/'+str(class_count[i]))
    
if __name__ == '__main__':
    main()

