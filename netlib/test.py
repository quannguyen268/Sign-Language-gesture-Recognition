import tensorflow as tf
import tf_slim as slim
from netutil import resnet_v1, resnet_utils
from netlib.basemodel import basenet
import numpy as np
tf = tf.compat.v1
tf.disable_v2_behavior()


x = tf.placeholder(tf.float32, shape=[None, 32,32,3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

fn = slim.l2_regularizer(1e-5)
fn0 = tf.no_regularizer
global_step = tf.Variable(0, trainable=False)
with tf.name_scope('Model'):
    with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=fn,
                        biases_regularizer=fn0, normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm],
                            is_training=True,
                            updates_collections=None,
                            decay=0.9,
                            center=True,
                            scale=True,
                            epsilon=1e-5):
            y_conv = basenet(x, kp=0.75, is_training=True, outdims=10)

#%%

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

x_test = x_test.reshape(-1, 1000, 32, 32)
y_test = y_test.reshape(-1, 1000, 10)

# x_train, x_test = x_train.astype('float32')/255.0, x_test.astype('float32')/255.0

#%%

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6, allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(50):
        for j in range(0,x_train.shape[0],128):
            x_batch = x_train[j: j + 128]
            y_batch = y_train[j: j + 128]

            if i % 10 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x: x_batch,y_:y_batch})

                print("step {}, training accuracy {}".format(i, train_accuracy))

            sess.run(train_step, feed_dict={x: x_batch, y_: y_batch})



        test_acc = np.mean([sess.run(accuracy, feed_dict={x: x_test[i],y_: y_test[i]})
                            for i in range(10)])

        print("test accuracy: {}".format(test_acc))


