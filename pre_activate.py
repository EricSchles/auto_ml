from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
import pandas as pd
import random

class ReluDense(tf.Module):
    def __init__(self, in_features, out_features, init_weights=None, name=None):
        super().__init__(name=name)
        if init_weights is None:
            self.init_weights = tf.random.normal([out_features, out_features])
        else:
            self.init_weights = init_weights
        self.w = tf.Variable(
            self.init_weights, name='w'
        )
        self.b = tf.Variable(
            tf.zeros([out_features]), name='b'
        )
    
    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)

class Dense(tf.Module):
    def __init__(self, in_features, out_features, init_weights=None, name=None):
        super().__init__(name=name)
        if init_weights is None:
            self.init_weights = tf.random.normal([out_features, out_features])
        else:
            self.init_weights = init_weights
        self.w = tf.Variable(
            self.init_weights, name='w'
        )
        self.b = tf.Variable(
            tf.zeros([out_features]), name='b'
        )
    
    def __call__(self, x):
        return tf.matmul(x, self.w) + self.b
    
class NeuralNet(Model):
    def __init__(self, X_in, X_out, optimizer, other_network=None, threshold=0, max_layers=4, learning_rate=0.9):
        super(NeuralNet, self).__init__()
        self.number_of_relu_dense = 0
        self.number_of_vanilla_dense = 0
        self.relu_layers = []
        self.dense_layers = []
        for index in range(max_layers):
            if other_network is None:
                self.relu_layers.append(ReluDense(X_out, X_out))
            else:
                dense = ReluDense(X_out, X_out, init_weights=other_network.relu_layers[index].init_weights)
                self.relu_layers.append(dense)
            self.number_of_relu_dense += 1
        for index in range(max_layers):
            if other_network is None:
                self.dense_layers.append(Dense(X_out, X_out))
            else:
                dense = Dense(X_out, X_out, init_weights=other_network.dense_layers[index].init_weights)
                self.dense_layers.append(dense)
            self.number_of_vanilla_dense += 1
        self.out = Dense(X_in, X_out)
        self.optimizer = optimizer
        self.tape = None
        self.learning_rate = learning_rate
        self.other_network = other_network
        self.threshold = threshold

    def call(self, x):
        for layer in self.relu_layers:
            x = layer(x)
        for layer in self.dense_layers:
            x = layer(x)
        x = self.out(x)
        return tf.reduce_mean(x, axis=1)

    def step(self, x, y):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            pred = self.call(x)
            if self.other_network:
                other_pred = self.other_network(x)
                if mse(pred, other_pred) < self.threshold:
                    return
            loss = mse(pred, y)
            self.tape = tape
        gradients = self.tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

def mse(x, y):
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float64)
    return tf.metrics.MSE(x, y)

if __name__ == '__main__':
    X = pd.read_csv("X.csv")
    y = np.load("y.npy")
    X = X.values
    X_val = pd.read_csv("X_val.csv")
    X_val = X_val.values
    y_val = np.load("y_val.npy")
    learning_rate = 0.9
    optimizer = tf.optimizers.Adam(learning_rate)
    other_nn = NeuralNet(X.shape[0], X.shape[1], optimizer)
    num_steps = 100
    for step in range(num_steps):
        other_nn.step(X, y)
        pred = other_nn(X_val)
        loss = mse(pred, y_val)

    print("mse", loss)
    nn = NeuralNet(X.shape[0], X.shape[1], optimizer, other_network=other_nn, threshold=0.1)
    num_steps = 100
    for step in range(num_steps):
        nn.step(X, y)
        pred = nn(X_val)
        loss = mse(pred, y_val)
    print("mse", loss)
