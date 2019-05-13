# -*- coding: utf-8 -*-
"""
Created on Sat May 11 12:56:47 2019

@author: hqc
"""

import tensorflow as tf
print(tf.__version__)
print(tf.test.is_gpu_available())
import glob
import os
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers
from PIL import Image
import numpy as np


def _read_data(dataset_dir):
    data_dir = os.path.join('./dataset/', dataset_dir)
    data_dir = data_dir + '/*.jpg'
    eye_data = glob.glob(data_dir)
    x = []
    for img in eye_data:
        img = Image.open(img)
        img = np.asarray(img)
        x.append(img)
    x = np.array(x)
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.expand_dims(x,axis=3)
    # print(x.shape, y.shape)
    return x

def preprocess(class1_dir, class2_dir):
    x1 = _read_data(class1_dir)
    y1 = tf.ones([x1.shape[0]], dtype=tf.int32)

    x2 = _read_data(class2_dir)
    y2 = tf.zeros([x2.shape[0]], dtype=tf.int32)

    x = tf.concat([x1, x2], axis=0)
    y = tf.concat([y1, y2], axis=0)
    return x, y

train_x, train_y = preprocess('closedEyes', 'openEyes')
test_x, test_y = preprocess('close_test', 'open_test')

# print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

train_db = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_db = train_db.shuffle(100000).batch(64)
test_db = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_db = test_db.batch(64).shuffle(100000)

sample = next(iter(train_db))
# print(sample[0].shape, sample[1].shape)


class Basenet(keras.Model):
    def __init__(self):
        super(Basenet, self).__init__()

        self.conv1 = layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.conv2 = layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.maxpool1 = layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')

        self.conv3 = layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.conv4 = layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.maxpool2 = layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')

        self.conv5 = layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.conv6 = layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.maxpool3 = layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')

        self.conv7 = layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.conv8 = layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        self.maxpool4 = layers.MaxPool2D(pool_size=[4, 4], strides=4, padding='same')

        self.dense1 = layers.Dense(256, activation=tf.nn.relu)
        self.dense2 = layers.Dense(128, activation=tf.nn.relu)
        self.dense3 = layers.Dense(2)

    def call(self, input, training=None):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool3(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.maxpool4(x)

        x = tf.reshape(x, [-1, 512])
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)


        return x


basenet = Basenet()
basenet.build(input_shape=(None, 24, 24, 1))
basenet.summary()

# tensorboard
log_dir = './model/'
summary_writer = tf.summary.create_file_writer(log_dir)


optimizer = optimizers.Adam(learning_rate=1e-4)
variable = basenet.trainable_variables

for epoch in range(5):
    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            logit = basenet(x)


            y_onehot = tf.one_hot(y, depth=2)

            loss = tf.losses.categorical_crossentropy(y_onehot, logit, from_logits=True)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, variable)
        optimizer.apply_gradients(zip(grads, variable))

        if step % 10 == 0:
            print(epoch, step, 'loss:', float(loss))
            basenet.save_weights('./model/weights.ckpt')
            print('saved weights. ')

    with summary_writer.as_default():
        tf.summary.scalar('loss', float(loss), step=epoch)

    total_num = 0
    total_correct = 0
    for x, y in test_db:
        logit = basenet(x)


        prob = tf.nn.softmax(logit, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)

        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)


        total_num += x.shape[0]
        total_correct += int(correct)
    acc = total_correct / total_num
    # print(total_num, total_correct)
    print('epoch:', epoch, 'acc:', acc)


del basenet

basenet = Basenet()
basenet.load_weights('./model/weights.ckpt')
total_num = 0
total_correct = 0
for x, y in test_db:
    logit = basenet(x)

    prob = tf.nn.softmax(logit, axis=1)
    pred = tf.argmax(prob, axis=1)
    pred = tf.cast(pred, dtype=tf.int32)

    correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
    correct = tf.reduce_sum(correct)

    total_num += x.shape[0]
    total_correct += int(correct)
acc = total_correct / total_num
print('acc:', acc)



