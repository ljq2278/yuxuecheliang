import tensorflow as tf
import MyModel_simple as model
from tensorflow import keras
import os
# from keras.preprocessing.image import ImageDataGenerator
from multiprocessing import Process, Queue
import pandas as pd
import cv2
import numpy as np
dataPath = 'F:/dataset/RainSnowUtility/'
batchsize = 4
volSize_w = 320
volSize_h = 240
classNum = 2
modelPath = 'model'
baselr = 0.001
def train(dataQueue):
    x = tf.placeholder(
        tf.float32, [
            batchsize,
            volSize_h,
            volSize_w,
            4
        ],
        name='x-input')

    y = tf.placeholder(
        tf.int32, [
            batchsize,
            volSize_h,
            volSize_w,
            classNum
        ],
        name='y-input')

    lrPL = tf.placeholder(
        tf.float32,
        name='lr')
    global_step = tf.Variable(0, trainable=False)
    myModel = model.vgg16seg(x, batchsize, classNum)
    getProbsOp = myModel.getProbsOp
    getLossOp = myModel.buildAndGetLossSimple(y)
    trainOp = myModel.buildOptmrAndGetTrainOp(lrPL, global_step)

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        myModel.load_weights("f:/model/vgg16_weights.npz", sess)
        ckpt = tf.train.get_checkpoint_state(modelPath)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('can not find checkpoint')
        count = 0
        lr = baselr
        while count < 2000001:
            count += 1
            [imgMats, lblMats, imgKey, flag] = dataQueue.get()
            if flag == 1:
                _, loss, prop, step = sess.run(
                    [trainOp, getLossOp,
                     getProbsOp, global_step],
                    feed_dict={
                        x: imgMats,
                        y: lblMats,
                        lrPL: lr
                    })
                print('train: ' + imgKey + ": loss %f " % loss + ' liverResMean: %f' % np.mean(prop[:, :, :, 1]))
                if step % 100 == 0:
                    saver.save(sess, modelPath + '/model', global_step=global_step)
                    print('save step %d suceess' % step)
            else:
                loss, prop = sess.run(
                    [getLossOp,
                     getProbsOp],
                    feed_dict={
                        x: imgMats,
                        y: lblMats,
                        lrPL: lr
                    })
                cv2.imshow('img_rgb',imgMats[0,:,:,0:3])
                cv2.imshow('img_thermal', imgMats[0,:,:,3:6])
                cv2.imshow('label_1', lblMats[0,:,:,1].astype(np.float))
                cv2.imshow('prop', prop[0, :, :, 1])
                # cv2.imshow('label_11', label_11)
                cv2.waitKey(1000)
                print('vali: ' + imgKey + ": loss %f" % loss)


def prepareDataThread(dataQueue):
    trainInfo = pd.read_csv(dataPath + '/train/train.csv', sep=',', low_memory=False)

    trainDataList_rgb = [dataPath+'train/'+itm for itm in trainInfo['rgb_path']]
    trainDataList_thermal = [dataPath + 'train/' + itm for itm in trainInfo['thermal_path']]
    trainLabelList = [dataPath+'train/'+itm for itm in trainInfo['mask_path']]

    order = np.random.randint(0,len(trainDataList_rgb),1000000)
    imgMats_rgb = np.zeros(shape=[batchsize, volSize_h, volSize_w, 3], dtype=np.float32)
    imgMats_thermal = np.zeros(shape=[batchsize, volSize_h, volSize_w, 1], dtype=np.float32)
    lblMats = np.zeros(shape=[batchsize, volSize_h, volSize_w, 2], dtype=np.int32)
    bi = 0
    for ind in order:
        img_rgb = cv2.imread(trainDataList_rgb[ind])
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb,(volSize_w,volSize_h))
        max_pix = np.max(img_rgb)
        img_rgb = img_rgb / max_pix

        img_thermal = cv2.imread(trainDataList_thermal[ind])
        img_thermal = cv2.cvtColor(img_thermal, cv2.COLOR_BGR2RGB)
        img_thermal = cv2.resize(img_thermal, (volSize_w,volSize_h))
        max_pix = np.max(img_thermal)
        min_pix = np.min(img_thermal)
        img_thermal = (img_thermal - min_pix) / (max_pix - min_pix)

        img_mask = cv2.imread(trainLabelList[ind])
        img_mask = cv2.resize(img_mask, (volSize_w,volSize_h))
        label_1 = (img_mask[:, :, 0] > 0).astype(float)
        # label_11 = img_mask[:, :, 0] / 255.0
        if np.sum(label_1) <= 4:
            continue
        label_0 = 1 - label_1
        label = np.zeros((img_mask.shape[0], img_mask.shape[1], 2), dtype=np.int)
        label[:, :, 0] = label_0
        label[:, :, 1] = label_1

        # cv2.imshow('img_rgb',img_rgb)
        # cv2.imshow('img_thermal', img_thermal)
        # cv2.imshow('label_1', label_1)
        # # cv2.imshow('label_11', label_11)
        # cv2.waitKey()

        imgMats_rgb[bi,:,:,:] = img_rgb
        imgMats_thermal[bi, :, :, 0] = img_thermal[:,:,0]
        lblMats[bi, :, :, :] = label
        if bi == batchsize-1:
            bi = 0
            dataQueue.put(tuple((np.concatenate([imgMats_rgb,imgMats_thermal],axis=3), lblMats, trainDataList_rgb[ind], 1)))
        else:
            bi += 1

def prepareDataThreadVali(dataQueue):
    testInfo = pd.read_csv(dataPath+'/test/test.csv',sep=',',low_memory=False)

    testDataList_rgb = [dataPath+'test/'+itm for itm in testInfo['rgb_path']]
    testDataList_thermal = [dataPath + 'test/' + itm for itm in testInfo['thermal_path']]
    testLabelList = [dataPath+'test/'+itm for itm in testInfo['mask_path']]

    order = np.random.randint(0,len(testDataList_rgb),1000000)
    imgMats_rgb = np.zeros(shape=[batchsize, volSize_h, volSize_w, 3], dtype=np.float32)
    imgMats_thermal = np.zeros(shape=[batchsize, volSize_h, volSize_w, 1], dtype=np.float32)
    lblMats = np.zeros(shape=[batchsize, volSize_h, volSize_w, 2], dtype=np.int32)
    bi = 0
    for ind in order:
        img_rgb = cv2.imread(testDataList_rgb[ind])
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (volSize_w,volSize_h))
        max_pix = np.max(img_rgb)
        img_rgb = img_rgb / max_pix

        img_thermal = cv2.imread(testDataList_thermal[ind])
        img_thermal = cv2.cvtColor(img_thermal, cv2.COLOR_BGR2RGB)
        img_thermal = cv2.resize(img_thermal, (volSize_w,volSize_h))
        max_pix = np.max(img_thermal)
        min_pix = np.min(img_thermal)
        img_thermal = (img_thermal-min_pix) / (max_pix-min_pix)

        img_mask = cv2.imread(testLabelList[ind])
        img_mask = cv2.resize(img_mask, (volSize_w,volSize_h))
        # label_1 = img_mask[:, :, 0] / 255.0
        label_1 = (img_mask[:, :, 0] > 0).astype(float)
        if np.sum(label_1) <= 4:
            continue
        label_0 = 1 - label_1
        label = np.zeros((img_mask.shape[0], img_mask.shape[1], 2), dtype=np.int)
        label[:, :, 0] = label_0
        label[:, :, 1] = label_1

        # cv2.imshow('img_rgb',img_rgb)
        # cv2.imshow('img_thermal', img_thermal)
        # cv2.imshow('label_1', label_1)
        # cv2.waitKey()

        imgMats_rgb[bi,:,:,:] = img_rgb
        imgMats_thermal[bi, :, :, 0] = img_thermal[:,:,0]
        lblMats[bi, :, :, :] = label
        if bi == batchsize-1:
            bi = 0
            if np.random.random() < 0.01:
                dataQueue.put(tuple((np.concatenate([imgMats_rgb,imgMats_thermal],axis=3), lblMats, testDataList_rgb[ind], 0)))
        else:
            bi += 1

if __name__ == '__main__':

    dataQueue = Queue(50)  # max 50 images in queue
    dataPreparation = [None] *2
    # print('params.params[ModelParams][nProc]: ', dataProdProcNum)
    # for proc in range(0, 1):
        # dataPreparation[proc] = Process(target=prepareDataThread, args=(dataQueue, numpyImages, numpyGT))
    dataPreparation[0] = Process(target=prepareDataThread, args=(dataQueue,))
    dataPreparation[0].daemon = True
    dataPreparation[0].start()
    dataPreparation[1] = Process(target=prepareDataThreadVali, args=(dataQueue,))
    dataPreparation[1].daemon = True
    dataPreparation[1].start()
    # while True:
    #     tt=1
    train(dataQueue)
