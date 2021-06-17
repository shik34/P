from __future__ import print_function
import shutil
import os.path
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

EXPORT_DIR = './model'

if os.path.exists(EXPORT_DIR):
    shutil.rmtree(EXPORT_DIR)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10


n_input = 784
n_classes = 10
                                                    dropout = 0.75


x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  #



def conv2d(x, W, b, strides=1):

    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):

    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')



def conv_net(x, weights, biases, dropout):

    x = tf.reshape(x, shape=[-1, 28, 28, 1])


    conv1 = conv2d(x, weights['wc1'], biases['bc1'])

    conv1 = maxpool2d(conv1, k=2)


    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])

    conv2 = maxpool2d(conv2, k=2)


    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.nn.dropout(fc1, dropout)


    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out



weights = {

    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),

    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),

    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),

    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


pred = conv_net(x, weights, biases, keep_prob)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)
    step = 1

    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:

            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")


    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                        y: mnist.test.labels[:256],
                                        keep_prob: 1.}))
    WC1 = weights['wc1'].eval(sess)
    BC1 = biases['bc1'].eval(sess)
    WC2 = weights['wc2'].eval(sess)
    BC2 = biases['bc2'].eval(sess)
    WD1 = weights['wd1'].eval(sess)
    BD1 = biases['bd1'].eval(sess)
    W_OUT = weights['out'].eval(sess)
    B_OUT = biases['out'].eval(sess)


g = tf.Graph()
with g.as_default():
    x_2 = tf.placeholder("float", shape=[None, 784], name="input")

    WC1 = tf.constant(WC1, name="WC1")
    BC1 = tf.constant(BC1, name="BC1")
    x_image = tf.reshape(x_2, [-1, 28, 28, 1])
    CONV1 = conv2d(x_image, WC1, BC1)
    MAXPOOL1 = maxpool2d(CONV1, k=2)

    WC2 = tf.constant(WC2, name="WC2")
    BC2 = tf.constant(BC2, name="BC2")
    CONV2 = conv2d(MAXPOOL1, WC2, BC2)
    MAXPOOL2 = maxpool2d(CONV2, k=2)

    WD1 = tf.constant(WD1, name="WD1")
    BD1 = tf.constant(BD1, name="BD1")

    FC1 = tf.reshape(MAXPOOL2, [-1, WD1.get_shape().as_list()[0]])
    FC1 = tf.add(tf.matmul(FC1, WD1), BD1)
    FC1 = tf.nn.relu(FC1)

    W_OUT = tf.constant(W_OUT, name="W_OUT")
    B_OUT = tf.constant(B_OUT, name="B_OUT")



    OUTPUT = tf.nn.softmax(tf.matmul(FC1, W_OUT) + B_OUT, name="output")

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    graph_def = g.as_graph_def()
    tf.train.write_graph(graph_def, EXPORT_DIR, 'mnist_model_graph.pb', as_text=False)


    y_train = tf.placeholder("float", [None, 10])
    correct_prediction = tf.equal(tf.argmax(OUTPUT, 1), tf.argmax(y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("check accuracy %g" % accuracy.eval(
            {x_2: mnist.test.images, y_train: mnist.test.labels}, sess))
