import tensorflow as tf
import numpy as np
from DesignFlags import DesignFlags
import matplotlib.pyplot as plt

design_flags = DesignFlags()

def generate_test_values():
    train_x = []
    train_y = []

    for _ in range(design_flags.m):
        x = np.random.rand()
        y = design_flags.slope * x + design_flags.y_int
        train_x.append([x])
        train_y.append(y)

    return np.array(train_x), np.transpose([train_y])

x = tf.placeholder(tf.float32, [None, 1], name="x")
W = tf.Variable(tf.zeros([1, 1]), name="W")
b = tf.Variable(tf.zeros([1]), name="b")
y = tf.placeholder(tf.float32, [None, 1])

model = tf.add(tf.matmul(x, W), b)

cost = tf.reduce_mean(tf.square(y - model))

train = tf.train.GradientDescentOptimizer(design_flags.learning_rate).minimize(cost)

train_dataset, train_values = generate_test_values()

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    for _ in range(design_flags.epochs):

        session.run(train, feed_dict={
            x: train_dataset,
            y: train_values
        })

    print("cost = {}".format(session.run(cost, feed_dict={
        x: train_dataset,
        y: train_values
    })))

    print("W = {}".format(session.run(W)))
    print("b = {}".format(session.run(b)))
    
    weight = session.run(W)
    y_int = session.run(b)

lin_model = weight * train_dataset + y_int
original_label = 'y_model = ' + str(design_flags.slope) + 'x + ' + str(design_flags.y_int)
lin_label = 'y_regression = ' + str(weight[0][0]) + 'x + ' + str(y_int[0])
plt.plot(train_dataset, train_values, label=original_label)
plt.plot(train_dataset, lin_model, label=lin_label)
plt.legend()
plt.show()