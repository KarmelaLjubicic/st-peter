import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from datetime import datetime
from tensorflow.contrib.framework import arg_scope as tf_arg_scope

num_features = 10
batch_size = tf.placeholder_with_default(tf.constant(10, dtype=tf.int64), shape=(), name="batch_size")
repeat_count = tf.placeholder_with_default(tf.constant(1, dtype=tf.int64), shape=(), name="repeat_count")

input_l_placeholder = tf.placeholder(tf.float32, [None, num_features*2], name="input_left_placeholder")
keep_prob = tf.placeholder_with_default(1.0, [])
input_keep_prob = tf.placeholder_with_default(1.0, [])

dataset = tf.data.Dataset.from_tensor_slices(input_l_placeholder)
dataset = dataset.repeat(repeat_count).shuffle(100000).batch(batch_size)
iterator = dataset.make_initializable_iterator()
batch_left = iterator.get_next()
batch_right = tf.concat([batch_left[:, num_features:], batch_left[:, :num_features]], axis=1)

l1_decay = tf.placeholder_with_default(tf.constant(0.0, dtype=tf.float32), shape=(), name="l1_decay")
initializer = tf.initializers.variance_scaling()


def create_model(inputs):
    with tf_arg_scope([tf_layers.fully_connected], activation_fn=tf.nn.relu, weights_initializer=initializer,
                      weights_regularizer=tf.contrib.layers.l1_regularizer(l1_decay), reuse=tf.AUTO_REUSE):
        layer_1 = tf_layers.fully_connected(inputs, 50, scope="layer_1")
        tf.summary.histogram("layer_1", layer_1, family="dense")
        layer_1 = tf.contrib.nn.alpha_dropout(layer_1, keep_prob)

        layer_2 = tf_layers.fully_connected(layer_1, 25, scope="layer_2")
        tf.summary.histogram("layer_2", layer_2, family="dense")
        layer_2 = tf.contrib.nn.alpha_dropout(layer_2, keep_prob)

        layer_3 = tf_layers.fully_connected(layer_2, 10, scope="layer_3")
        tf.summary.histogram("layer_3", layer_3, family="dense")
        layer_3 = tf.contrib.nn.alpha_dropout(layer_3, keep_prob)

        outputs = tf_layers.fully_connected(layer_3, 1, activation_fn=None, scope="outputs")

    tf.summary.histogram("outputs", outputs, family="dense")
    return outputs


############################################################
#   U lijevu mrezu dolaze samo dobri primjeri (batch_good), u desnu samo losi (batch_bad).
#   Isprobala sam dva moguca losa: cross_entropy i suma sigm(logit)


with tf.variable_scope("model"):
    f_left = create_model(batch_left)

with tf.variable_scope("model", reuse=True):
    f_right = create_model(batch_right)

with tf.variable_scope("loss"):
    logit = f_left - f_right
    h = tf.sigmoid(logit)
    y = tf.fill(tf.shape(logit), 1)
    loss = tf.losses.sigmoid_cross_entropy([[1]], logit) + tf.losses.get_regularization_loss()
    #loss = - tf.reduce_sum(h)

learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver(max_to_keep=10000)
sess = tf.Session()

timestamp = "{:%d-%m-%Y_%H-%M-%S}".format(datetime.now())
train_writer = tf.summary.FileWriter("./logdir/train_{0}".format(timestamp), graph=sess.graph, session=sess)
test_writer = tf.summary.FileWriter("./logdir/test_{0}".format(timestamp), graph=sess.graph, session=sess)

summary = tf.summary.merge_all()
