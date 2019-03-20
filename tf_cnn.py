import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('data/MNIST/', one_hot=True)
# define the hyper parameters
num_batches = 1000
batch_size = 50
learning_rate = 1e-4


# create a cnn model
class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # create layers of cnn
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64, ))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        # execute the model
        inputs = tf.reshape(inputs, [-1, 28, 28, 1])
        out = self.conv1(inputs)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.flatten(out)
        out = self.dense1(out)
        out = self.dense2(out)
        return out

    def predict(self, inputs):
        # predict the label of data
        logits = self(inputs)
        return logits


model = CNN()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

for iter in range(num_batches):
    X, y = None, None
    with tf.GradientTape() as tape:
        y_logit_pred = model(tf.convert_to_tensor(X))
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=y, logits=y_logit_pred)
        print("batch %d: loss %f" % (iter, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
