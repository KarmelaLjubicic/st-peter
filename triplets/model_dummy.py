import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from datetime import datetime
from tensorflow.contrib.framework import arg_scope as tf_arg_scope

num_features = 10
batch_size = tf.placeholder_with_default(tf.constant(10, dtype=tf.int64), shape=(), name="batch_size")
repeat_count = tf.placeholder_with_default(tf.constant(1, dtype=tf.int64), shape=(), name="repeat_count")

input_l_placeholder = tf.placeholder(tf.float32, [None, num_features*2], name="input_left_placeholder")
cnt = tf.placeholder(tf.int32, shape=())
A = tf.placeholder(tf.float32, name="A")
A = tf.reshape(A, [batch_size, cnt])
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
        #layer_1 = tf_layers.fully_connected(inputs, 50, scope="layer_1")
        #tf.summary.histogram("layer_1", layer_1, family="dense")
        #layer_1 = tf.contrib.nn.alpha_dropout(layer_1, keep_prob)

        layer_2 = tf_layers.fully_connected(inputs, 45, scope="layer_2")
        tf.summary.histogram("layer_2", layer_2, family="dense")
        #layer_2 = tf.contrib.nn.alpha_dropout(layer_2, keep_prob)

        layer_3 = tf_layers.fully_connected(layer_2, 25, scope="layer_3")
        tf.summary.histogram("layer_3", layer_3, family="dense")
        #layer_3 = tf.contrib.nn.alpha_dropout(layer_3, keep_prob)

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

    #A_t = tf.linalg.transpose(A)
    #X = tf.linalg.solve(tf.linalg.matmul(A_t, A) + [1], tf.linalg.matmul(A_t, h))
    #i_f = tf.norm(tf.matmul(A, X)) / tf.norm(h)

    # ----------------------------------------------------------
        # reg = ||l1 - l2| - l3|, gdje su l1 i l2 logit vrijednosti dobar-los parova koje je mreza već vidjela,
        # a l3 je logit za istovrsni par u trojci (npr ako je trojka od 2 dobra i jednog loseg, l3 je logit kad se mrezi
        # preda par dobar-dobar

    rem = tf.range(30000) % 3

    reg = tf.abs(tf.abs(tf.gather(logit, tf.reshape(tf.where(tf.equal(0, rem)), (-1,1))) -
                        tf.gather(logit, tf.reshape(tf.where(tf.equal(1, rem)), (-1,1))))
                        - tf.gather(logit, tf.reshape(tf.where(tf.equal(2, rem)), (-1,1))))
    loss = tf.losses.sigmoid_cross_entropy([[1]], logit) #* 1/i_f #+ tf.losses.get_regularization_loss()

learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
train_op_1 = tf.train.AdamOptimizer(learning_rate).minimize(loss + reg)

saver = tf.train.Saver(max_to_keep=10000)
sess = tf.Session()

timestamp = "{:%d-%m-%Y_%H-%M-%S}".format(datetime.now())
train_writer = tf.summary.FileWriter("./logdir/train_{0}".format(timestamp), graph=sess.graph, session=sess)
test_writer = tf.summary.FileWriter("./logdir/test_{0}".format(timestamp), graph=sess.graph, session=sess)

summary = tf.summary.merge_all()
