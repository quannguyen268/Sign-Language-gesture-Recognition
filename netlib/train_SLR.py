import tensorflow as tf
import tf_slim as slim
from netlib.basemodel import basenet, basenet3
import numpy as np
import cv2

tf = tf.compat.v1
tf.disable_v2_behavior()
rng = np.random.RandomState(23455)


# %%

def load_batch_data(list_path_image, batch_size=128, index=0):
    batch_data_path = list_path_image[index: index + batch_size]
    images = []
    labels = []
    for i in batch_data_path:
        image_name = i.split('/')[-1]
        label = int(image_name.split('_')[0]) - 1
        image = cv2.imread(i, 0)
        if image.shape != (96, 96):
            image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
        image = np.expand_dims(image, axis=-1)
        # label = np.expand_dims(label, axis=-1)

        images.append(image)
        labels.append(label)
    return np.asarray(images), np.asarray(labels)


# %%

DATA_PATH = '/home/quan/PycharmProjects/SLR/filename.txt'
f = open(DATA_PATH, 'r')

list_data = f.readlines()  # list image truyền trước ở path_to_data
list_data = [path.split('\n')[0] for path in list_data]
# rng.shuffle(list_data)
list_img = [path.split(" ")[0] for path in list_data]
list_distance = [path.split(" ")[1] for path in list_data]

list_img_train = list_img[1000:]
# rng.shuffle(list_img_train)
list_distance_train = list_distance[1000:]

list_img_test = list_img[:1000]

list_distance_test = list_distance[:1000]

print(len(list_img_train), len(list_img_test))
print(list_distance_train[:10])
# %%
# Add some ops that take care of logging summaries
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
        tf.Variable

with tf.name_scope('learning_rate'):
    lr = tf.Variable((1e-3), dtype=tf.float32, trainable=False)
    variable_summaries(lr)
    tf.summary.scalar("learing_rate", lr)

x = tf.placeholder(tf.float32, shape=[None, 96, 96, 1])
y_ = tf.placeholder(tf.int32, shape=[None, ])
d = tf.placeholder(tf.float32, shape=[None, 1])

# %%
print(x.name)
# %%

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
            y_pred = basenet3(x, d, kp=0.75, is_training=True, outdims=64)


#%%

y_pred.get_shape()
# %%
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y_pred))
    tf.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy, global_step)
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_pred, axis=0), tf.argmax(y_, axis=0))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100
    tf.summary.scalar('accuracy', accuracy)

summ = tf.summary.merge_all()
saver = tf.train.Saver()


# %%


LOG_DIR = '/home/quan/PycharmProjects/SLR/logs/'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7, allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
    # Write summaries to LOG_DIR -- used by Tensorboard

    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test', graph=tf.get_default_graph())
    init = tf.global_variables_initializer()

    sess.run(init)

    train_num = len(list_img_train)
    test_num = len(list_img_test)
    batch_size = 16
    steps_per_epoch = train_num // batch_size
    idx_train = 0
    idx_test = 0
    step = 0
    epochs = 100

    for itrain in range(steps_per_epoch * epochs):
        if itrain % steps_per_epoch == 0:
            lr_update = tf.assign(lr, (1e-3) * 0.96 ** step)
            sess.run(lr_update)
            step += 1

        batch_dist_train = list_distance_train[idx_train: idx_train + batch_size]
        batch_dist_train = np.reshape(batch_dist_train, (-1, 1))
        batch_image, batch_label = load_batch_data(list_img_train, batch_size=batch_size, index=idx_train)
        idx_train = (idx_train + batch_size) % train_num
        if (idx_train + batch_size) >= train_num:
            idx_train = idx_train + batch_size - train_num

        batch_image = batch_image.astype('float32') / 127.5 - 1.
        batch_image = np.reshape(batch_image, (-1, 96, 96, 1))
        # x=tf.get_default_graph().get_tensor_by_name("Placeholder:0")
        summary, _, step = sess.run([summ, train_step, global_step], feed_dict={x: batch_image,
                                                                          d: batch_dist_train,
                                                                          y_: batch_label})
        # train_writer.add_summary(summary, step)

        if itrain % steps_per_epoch == 0:
            pred_acc = []
            pred_loss = []
            loop = 32 // batch_size
            for itest in range(loop + 1):
                batch_dist_test = list_distance_test[idx_test: idx_test + batch_size]
                batch_dist_test = np.reshape(batch_dist_test, (-1, 1))

                batch_image_test, batch_label_test = load_batch_data(list_img_test, batch_size=batch_size, index=idx_test)
                idx_test = (idx_test + batch_size) % test_num
                if (idx_test + batch_size) >= test_num:
                    idx_test = idx_test + batch_size - test_num
                batch_image_test = batch_image_test.astype('float32') / 127.5 - 1.
                batch_image_test = np.reshape(batch_image_test, (-1, 96, 96, 1))
                summary, acc, loss, = sess.run([summ, accuracy, cross_entropy], feed_dict={x: batch_image_test,
                                                                                           d: batch_dist_test,
                                                                                           y_: batch_label_test})

                pred_acc.append(acc)
                pred_loss.append(loss)

            loss = np.mean(pred_loss)
            acc = np.mean(acc)
            print("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            #
            # test_writer.add_summary(summary, step)
            #
            # logt = open('/home/quan/PycharmProjects/SLR/logs/logt_LSA64_{}.txt'.format(0), 'a+')
            # logt.write('epoch: {}, test accuracy: {}, test loss: {}'.format(step, acc, loss))
            # logt.write('\n')
            # logt.close()
            #
            # saver.save(sess, '/home/quan/PycharmProjects/SLR/Model/SLR_LSA64_{}'.format(acc), global_step=step)

# %%
layers_per_block = [2, 2, 4, 4]
out_chan_list = [64, 128, 256, 512]
pool_list = [True, True, True, False]

# learn some feature representation, that describes the image content well
for block_id, (layer_num, chan_num, pool) in enumerate(zip(layers_per_block, out_chan_list, pool_list), 1):
    print(block_id, layer_num, chan_num, pool)
    # for layer_id in range(layer_num):
    # if pool:
