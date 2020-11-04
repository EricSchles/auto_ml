from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
import pandas as pd
import random

class ReluDense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(
            tf.random.normal([out_features, out_features]), name='w'
        )
        self.b = tf.Variable(
            tf.zeros([out_features]), name='b'
        )
    
    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)

class Dense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(
            tf.random.normal([out_features, out_features]), name='w'
        )
        self.b = tf.Variable(
            tf.zeros([out_features]), name='b'
        )
    
    def __call__(self, x):
        return tf.matmul(x, self.w) + self.b
    
class NeuralNet(Model):
    def __init__(self, X_in, X_out, optimizer, dropout=0.1):
        super(NeuralNet, self).__init__()
        self.number_of_relu_dense = 0
        self.number_of_vanilla_dense = 0
        self.relu_layers = []
        self.dense_layers = []
        for _ in range(1, random.randint(2, 10)):
            self.relu_layers.append(ReluDense(X_in, X_out))
            self.number_of_relu_dense += 1
        for _ in range(1, random.randint(2, 10)):
            self.dense_layers.append(Dense(X_in, X_out))
            self.number_of_vanilla_dense += 1
        self.out = Dense(X_in, X_out)
        self.optimizer = optimizer
        self.dropout = dropout
        self.tape = None

    def call(self, x, train=False):
        if train:
            for layer in self.relu_layers:
                x = layer(x)
                x = dropout(x, self.dropout)
            for layer in self.dense_layers:
                x = layer(x)
                x = dropout(x, self.dropout)
        else:
            for layer in self.relu_layers:
                x = layer(x)
                x = dropout(x, self.dropout)
            for layer in self.dense_layers:
                x = layer(x)
                x = dropout(x, self.dropout)
            x = self.out(x)
        return tf.reduce_mean(x, axis=1)

    def step(self, x, y):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            pred = self.call(x, train=True)
            loss = mse(pred, y)
            self.tape = tape
            
        gradients = self.tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

# naive strategy - could also average weights and bias term instead
class EnsembleNet(Model):
    def __init__(self, net_one, net_two, weight):
        super(EnsembleNet, self).__init__()
        self.net_one = net_one
        self.net_two = net_two
        # weight is two numbers between 0 and 1
        # e.g. [0.5, 0.5]
        # this is the weight on the result of each network
        self.weight = weight
        
    def call(self, x, train=False):
        x_one = self.net_one(x, train=train)
        x_two = self.net_two(x, train=train)
        return x_one * self.weight[0] + x_two * self.weight[1]

    def step(self, x, y):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        pred = self.call(x, train=True)
        loss = mse(pred, y)

        gradients_one = self.net_one.tape.gradient(loss, self.net_one.trainable_variables)
        self.net_one.optimizer.apply_gradients(zip(gradients_one, self.net_one.trainable_variables))
        gradients_two = self.net_two.tape.gradient(loss, self.net_two.trainable_variables)
        self.net_two.optimizer.apply_gradients(zip(gradients_two, self.net_two.trainable_variables))
    
def dropout(X, drop_probability):
    keep_probability = 1 - drop_probability
    mask = np.random.uniform(0, 1.0) < keep_probability
    if keep_probability > 0.0:
        scale = (1/keep_probability)
    else:
        scale = 0.0
    return mask * X * scale

def mse(x, y):
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float64)
    return tf.metrics.MSE(x, y)

def maape(y, y_pred):
    normed_absolute_error = np.abs(y_pred - y) / y
    normed_arctan_abs_error = np.arctan(normed_absolute_error)
    return np.sum(normed_arctan_abs_error)

def train_nets(X, y, neural_nets, num_steps):
    stopped = []
    mse_losses = [[] for _ in range(len(neural_nets))]
    maape_losses = [[] for _ in range(len(neural_nets))]
    for step in range(num_steps):
        for index in range(len(neural_nets)):
            if index in stopped:
                continue
            nn = neural_nets[index]
            nn.step(X, y)
            pred = nn(X_val)
            loss_mse = mse(pred, y_val)
            loss_maape = maape(pred, y_val)
            is_nan = pd.isnull(loss_mse.numpy()) or pd.isnull(loss_maape)
            is_less_than_zero = loss_mse.numpy() < 0 or loss_maape < 0
            if is_nan or is_less_than_zero:
                stopped.append(index)
                continue
            mse_losses[index].append(loss_mse)
            maape_losses[index].append(loss_maape)

    neural_nets = [
        neural_nets[index]
        for index in range(len(neural_nets))
        if index not in stopped
    ]
    mse_losses = [
        mse_losses[index]
        for index in range(len(mse_losses))
        if index not in stopped
    ]
    maape_losses = [
        maape_losses[index]
        for index in range(len(maape_losses))
        if index not in stopped
    ]
    return mse_losses, maape_losses, neural_nets

def select_best_k_networks(losses, k, networks):
    average_loss = []
    last_25_percent_index = int(len(losses[0]) * 0.75)
    for index, loss in enumerate(losses):
        last_25_percent = loss[last_25_percent_index:]
        average_loss.append(
            [index, np.mean(last_25_percent)]
        )
    average_loss = sorted(average_loss, key=lambda t: t[1])
    average_loss_index = [ave_loss[0] for ave_loss in average_loss[:k]]
    return [
        networks[index] for index in average_loss_index
    ]
    
        
        
if __name__ == '__main__':
    X = pd.read_csv("X.csv")
    y = np.load("y.npy")
    X = X.values
    X_val = pd.read_csv("X_val.csv")
    X_val = X_val.values
    y_val = np.load("y_val.npy")
    flip_maape = False
    mated_nets = []
    learning_rate = 0.9
    optimizer = tf.optimizers.Adam(learning_rate)
    num_steps = 2
    k_best = 10
    for _ in range(3):
        neural_nets = [NeuralNet(X.shape[0], X.shape[1], optimizer)
                       for _ in range(50)]

        mse_losses, maape_losses, neural_nets = train_nets(X, y, neural_nets, num_steps)
        best_mse_nets = select_best_k_networks(mse_losses, k_best, neural_nets)
        best_maape_nets = select_best_k_networks(maape_losses, k_best, neural_nets)
        
        if flip_maape:
            best_maape_nets = best_maape_nets[::-1]
        for index in range(len(best_mse_nets)):
            ensemble = EnsembleNet(best_mse_nets[index], best_maape_nets[index], [0.5, 0.5])
            mated_nets.append(ensemble)

    mse_losses, maape_losses, neural_nets = train_nets(X, y, mated_nets, num_steps)    
    min_mses = []
    min_maapes = []
    for mse_loss in mse_losses:
        min_mses.append(min(mse_loss))
    for maape_loss in maape_losses:
        min_maapes.append(min(maape_loss))
    print("mse", min(min_mses))
    print("maape", min(min_maapes))
