import numpy as np
import tensorflow as tf
# import global_variable
# import tflearn as tfl
# from tensorflow.contrib.slim.nets import resnet_v2
# from dropBlock.dropblock import DropBlock

class vgg16seg:

    def __init__(self, imgs,batchsize,classNum):
        self.batchsize = batchsize
        self.classNum = classNum
        self.parameters = []
        self.showout = []
        self.imgs = imgs
        self.convlayers_ori()
        # self.fc_layers()
        self.getProbsOp = tf.nn.softmax(self.conv9_4)
        self.showout.append(self.getProbsOp)


    def saver(self):
        return tf.train.Saver()

    def maxpool(self,name,input_data):
        out = tf.nn.max_pool(input_data,[1,2,2,1],[1,2,2,1],padding="SAME",name=name)
        return out

    def conv(self,name, ksize,input_data, out_channel,trainable,restrain=True):
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [ksize, ksize, in_channel, out_channel], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),trainable=trainable)
            biases = tf.get_variable("biases", [out_channel], initializer=tf.constant_initializer(0.0),dtype=tf.float32,trainable=trainable)
            conv_res = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding="SAME")
            res = tf.nn.bias_add(conv_res, biases)
            if restrain==True:
                out = tf.layers.batch_normalization(tf.nn.relu(res, name=name))
            else:
                out = res
        self.parameters += [kernel, biases]
        return out

    def conv_softmax(self,name, ksize,input_data, out_channel,trainable,restrain=True):
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [ksize, ksize, in_channel, out_channel], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),trainable=trainable)
            biases = tf.get_variable("biases", [out_channel], initializer=tf.constant_initializer(0.0),dtype=tf.float32,trainable=trainable)
            conv_res = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding="SAME")
            res = tf.nn.bias_add(conv_res, biases)
            out = tf.layers.batch_normalization(tf.nn.relu(res, name=name))
            s_out = tf.nn.softmax(out)
        self.parameters += [kernel, biases]
        return out, s_out

    def upconv(self,name, input_data, out_channel,trainable):
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [2, 2, out_channel, in_channel], dtype=tf.float32,trainable=trainable)
            biases = tf.get_variable("biases", [out_channel], dtype=tf.float32,trainable=trainable)

            conv_res = tf.nn.conv2d_transpose(input_data, kernel, output_shape = [self.batchsize,int(input_data.shape[1])*2,int(input_data.shape[2])*2,out_channel],strides = [1, 2, 2, 1], padding="SAME")
            res = tf.nn.bias_add(conv_res, biases)
            out = tf.layers.batch_normalization(tf.nn.relu(res, name=name))
        self.parameters += [kernel, biases]
        return out

    def upconv_softmax(self,name, input_data, out_channel,trainable):
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [5, 5, out_channel, in_channel], dtype=tf.float32,trainable=trainable)
            biases = tf.get_variable("biases", [out_channel], dtype=tf.float32,trainable=trainable)

            conv_res = tf.nn.conv2d_transpose(input_data, kernel, output_shape = [self.batchsize,int(input_data.shape[1])*2,int(input_data.shape[2])*2,out_channel],strides = [1, 2, 2, 1], padding="SAME")
            res = tf.nn.bias_add(conv_res, biases)
            out = tf.layers.batch_normalization(tf.nn.relu(res, name=name))
            s_out = tf.nn.softmax(out)
        self.parameters += [kernel, biases]
        return out,s_out

    def conct(self,tensor1,tensor2):
        layerConc = tf.concat([tensor1, tensor2], 3)
        return layerConc

    def pontWiseAdd(self,tensor1,tensor2):
        ret = tf.layers.batch_normalization(tf.nn.relu(tensor1 + tensor2))
        return ret

    def convlayers_ori(self):
        # zero-mean input
        #conv1
        self.conv1_1 = self.conv("conv1re_1",3,self.imgs,64,trainable=False) #1
        self.conv1_2 = self.conv("conv1_2",3,self.conv1_1,64,trainable=False)#1
        self.pool1 = self.maxpool("poolre1",self.conv1_2) #1/2

        #conv2
        self.conv2_1 = self.conv("conv2_1",3,self.pool1,128,trainable=False)#1/2
        self.conv2_2 = self.conv("convwe2_2",3,self.conv2_1,128,trainable=False)#1/2
        self.pool2 = self.maxpool("pool2",self.conv2_2)#1/4

        #conv3
        self.conv3_1 = self.conv("conv3_1",3,self.pool2,256,trainable=False)#1/4
        self.conv3_2 = self.conv("convrwe3_2",3,self.conv3_1,256,trainable=False)#1/4
        self.conv3_3 = self.conv("convrew3_3",3,self.conv3_2,256,trainable=False)#1/4
        self.pool3 = self.maxpool("poolre3",self.conv3_3)#1/8

        #conv4
        self.conv4_1 = self.conv("conv4_1",3,self.pool3,512,trainable=False)#1/8
        self.conv4_2 = self.conv("convrwe4_2",3,self.conv4_1,512,trainable=False)#1/8
        self.conv4_3 = self.conv("conv4rwe_3",3,self.conv4_2,512,trainable=False)#1/8

        # upconv2
        self.upconv2 = self.upconv("upconv2", self.conct(self.conv4_3,self.pool3), 256,trainable=True)#1/4

        # conv7
        self.conv7_1 = self.conv("conv7_1", 3,self.upconv2, 256,trainable=True)#1/4
        self.conv7_2 = self.conv("conv7_2", 3,self.conv7_1, 256,trainable=True)#1/4
        self.conv7_3 = self.conv("conv7_3", 3,self.conv7_2, 256,trainable=True)#1/4

        # upconv3
        self.upconv3 = self.upconv("upconv3", self.conct(self.conv7_3,self.pool2), 128,trainable=True)#1/2

        # conv8
        self.conv8_1 = self.conv("conv8_1", 3,self.upconv3, 128,trainable=True)#1/2
        self.conv8_2 = self.conv("conv8_2", 3,self.conv8_1, 128,trainable=True)#1/2

        # upconv4
        self.upconv4 = self.upconv("upconv4", self.conct(self.conv8_2, self.pool1), 64,trainable=True)  # 1

        # conv9
        self.conv9_1 = self.conv("conv9_1", 3,self.upconv4, 32,trainable=True)  # 1
        self.conv9_2 = self.conv("conv9_2", 3,self.conv9_1, 16,trainable=True)  # 1

        self.conv9_3 = self.conv("conv9_3", 3,self.conct(self.conv9_2, self.imgs), self.classNum,trainable=True)  #1
        self.conv9_4 = self.conv("conv9_4", 1, self.conv9_3, self.classNum, trainable=True,restrain=False)  # 1

    # def convlayers(self):
    #     # zero-mean input
    #     #conv1
    #     # imgs,thermals = tf.split(self.imgs, [3, 1], axis=3)
    #     self.conv1_1 = self.conv("conv1re_1", 3, self.imgs, 64, trainable=False)
    #     # self.conv1_1_2 = self.conv("conv1re_2", 3, thermals, 64, trainable=True)
    #     # self.conv1_1 = self.conv1_1_1+self.conv1_1_2 #1
    #     self.conv1_2 = self.conv("conv1_2",3,self.conv1_1,64,trainable=False)#1
    #     self.pool1 = self.maxpool("poolre1",self.conv1_2) #1/2
    #
    #     #conv2
    #     self.conv2_1 = self.conv("conv2_1",3,self.pool1,128,trainable=False)#1/2
    #     self.conv2_2 = self.conv("convwe2_2",3,self.conv2_1,128,trainable=False)#1/2
    #     self.pool2 = self.maxpool("pool2",self.conv2_2)#1/4
    #
    #     #conv3
    #     self.conv3_1 = self.conv("conv3_1",3,self.pool2,256,trainable=False)#1/4
    #     self.conv3_2 = self.conv("convrwe3_2",3,self.conv3_1,256,trainable=False)#1/4
    #     self.conv3_3 = self.conv("convrew3_3",3,self.conv3_2,256,trainable=False)#1/4
    #     self.pool3 = self.maxpool("poolre3",self.conv3_3)#1/8
    #
    #     #conv4
    #     self.conv4_1 = self.conv("conv4_1",3,self.pool3,512,trainable=False)#1/8
    #     self.conv4_2 = self.conv("convrwe4_2",3,self.conv4_1,512,trainable=False)#1/8
    #     self.conv4_3 = self.conv("conv4rwe_3",3,self.conv4_2,512,trainable=False)#1/8
    #     # self.pool4 = self.maxpool("pool4",self.conv4_3)#1/16
    #
    #     self.conv4_4, self.prop_c = self.conv_softmax("conv4_4", 5, self.conv4_2, self.classNum, trainable=True)  # 1/8
    #
    #
    #     self.upconv2, self.prop_b = self.upconv_softmax("upconv2", self.conct(self.prop_c,self.pool3), self.classNum, trainable=True)#1/4
    #
    #
    #     self.upconv3, self.prop_a = self.upconv_softmax("upconv3", self.conct(self.prop_b,self.pool2), self.classNum,trainable=True)#1/2
    #
    #
    #     self.upconv4, self.prop = self.upconv_softmax("upconv4", self.conct(self.prop_a, self.pool1), self.classNum,trainable=True)  # 1
    #
    #     # conv9
    #     # self.conv9_1 = self.conv("conv9_1", 3,self.upconv4, 32,trainable=True)  # 1
    #     # self.conv9_2 = self.conv("conv9_2", 3,self.conv9_1, 16,trainable=True)  # 1
    #     #
    #     # self.conv9_3 = self.conv("conv9_3", 3,self.conct(self.conv9_2, self.imgs), self.classNum,trainable=True)  #1
    #     # self.conv9_4 = self.conv("conv9_4", 1, self.conv9_3, self.classNum, trainable=True,restrain=False)  # 1


    # def buildAndGetLossSimple(self,label):
    #     self.loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=self.upconv4, labels=label)
    #     label0, label1 = tf.split(tf.cast(label,tf.float32), [1, 1], axis=3)
    #
    #     label_a_1 = tf.cast(tf.nn.avg_pool(label1,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME')>0.5,tf.int32)
    #     label_a_0 = -(label_a_1 - 1)
    #     label_a = tf.concat([label_a_0,label_a_1],axis=3)
    #     self.loss_a = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.upconv3, labels=label_a))
    #
    #     label_b_1 = tf.cast(tf.nn.avg_pool(label1, (1,4, 4,1), (1,4, 4,1), 'SAME') > 0.5, tf.int32)
    #     label_b_0 = -(label_b_1 - 1)
    #     label_b = tf.concat([label_b_0, label_b_1], axis=3)
    #     self.loss_b = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.upconv2, labels=label_b))
    #
    #     label_c_1 = tf.cast(tf.nn.avg_pool(label1, (1,8, 8,1), (1,8, 8,1), 'SAME') > 0.5, tf.int32)
    #     label_c_0 = -(label_c_1 - 1)
    #     label_c = tf.concat([label_c_0, label_c_1], axis=3)
    #     self.loss_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.conv4_4, labels=label_c))
    #
    #     self.loss = tf.reduce_mean(self.loss_map)+2*self.loss_a+4*self.loss_b+8*self.loss_c
    #     return self.loss

    def buildAndGetLossWeighted(self,label):
        self.loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=self.conv9_4, labels=label)
        label0,label1 = tf.split(label,[1,1],axis=3)
        weight = label0+label1*8
        self.loss = tf.reduce_mean(self.loss_map*tf.cast(tf.squeeze(weight),tf.float32))
        return self.loss

    # def buildAndGetLoss(self,label):
    #     self.loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=self.conv9_4, labels=label)
    #     self.loss_mid = tf.reduce_mean(self.loss_map,[1,2])
    #
    #     num_examples = tf.shape(label)[0]
    #     n_selected = num_examples / 2  # 根据个人喜好只选最难的一半
    #
    #     n_selected = tf.cast(n_selected, tf.int32)
    #     vals, _ = tf.nn.top_k(self.loss_mid, k=n_selected)
    #     th = vals[-1]
    #     selected_mask = self.loss_mid >= th
    #
    #     loss_weight = tf.cast(selected_mask, tf.float32)
    #     self.loss = tf.reduce_sum(self.loss_mid * loss_weight) / tf.reduce_sum(loss_weight)
    #
    #     return self.loss

    # def buildAndGetLossTest(self,label):
    #     self.loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=self.conv9_4, labels=label)
    #     self.loss = tf.reduce_mean(self.loss_map)
    #     return self.loss

    # self.lr = tf.maximum(1e-5, tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
    #                                                       self.decay, staircase=True))

    def buildOptmrAndGetTrainOp(self,lr,global_step):
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss, global_step=global_step)
        return self.train_op


    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i not in [20,21,22,23,24,25,26,27,28,29,30,31]:
                sess.run(self.parameters[i].assign(weights[k]))
        print("-----------all done---------------")